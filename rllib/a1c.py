
import copy
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from rllib.basic import prefix, Data
from rllib.buffer import RolloutBuffer
from rllib.utils import init_weights, hard_update
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper


class A1C(MethodSingleAgent):
    """
        A2C with single worker.
    """
    def __init__(self, config, writer):
        super().__init__(config, writer)

        ### param
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.9)

        self.buffer_size = config.get('buffer_size', 2000)
        self.batch_size = self.buffer_size

        self.lr = config.get('lr', 0.0003)
        self.betas = config.get('betas', (0.9, 0.999))
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        self.epsilon_clip = config.get('epsilon_clip', 0.2)
        self.weight_value = config.get('weight_value', 0.5)
        self.weight_entropy = config.get('weight_entropy', 0.001)

        ### model
        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config).to(self.device)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.critic_loss = nn.MSELoss()

        self.buffer: RolloutBuffer = config.get('buffer', RolloutBuffer)(config, self.device, batch_size=-1, use_gae=True)


    def update_parameters(self):
        if len(self.buffer) < self.buffer_size:
            return
        self.update_parameters_start()
        print(prefix(self) + 'update step: ', self.step_update)


        experience = self.buffer.sample(self.gamma, self.gae_lambda, advantage_normalization=True)
        state = experience.state
        action = experience.action_data.action
        returns = experience.returns
        advantage = experience.advantage

        logprob, value, entropy, _ = self.policy.evaluate(state, action)

        loss_policy = (-logprob * advantage).mean()
        loss_value = self.weight_value* self.critic_loss(value, returns)
        loss_entropy = -self.weight_entropy* entropy.mean()
        loss = loss_policy + loss_entropy + loss_value

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.writer.add_scalar('method/loss', loss.detach().item(), self.step_update)
        self.writer.add_scalar('method/loss_surr', loss_policy.detach().item(), self.step_update)
        self.writer.add_scalar('method/loss_entropy', loss_entropy.detach().item(), self.step_update)
        self.writer.add_scalar('method/loss_value', loss_value.detach().item(), self.step_update)

        self.update_callback(locals())

        self._save_model()

        self.buffer.clear()
        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        action = self.policy(state.to(self.device))
        return action.cpu()




class ActorCriticDiscrete(Model):
    """
        config: net_ac_fe, net_actor_fm, net_critic_fm
    """
    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.fe = config.get('net_ac_fe', FeatureExtractor)(config, 0)
        self.actor = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.actor_no = nn.Softmax(dim=1)
        self.critic = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature, 1)
        self.apply(init_weights)


    def forward(self, state):
        x = self.fe(state)
        action_prob = self.actor_no(self.actor(x))
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.unsqueeze(1), action_prob, dist

    def evaluate(self, state, action):
        raise NotImplementedError
        x = self.fe(state)
        action_prob = self.actor_no(self.actor(x))
        value = self.critic(x)

        dist = Categorical(action_prob)
        logprob = dist.log_prob(action.squeeze()).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)

        return logprob, value, entropy, dist



class ActorCriticContinuous(Model):
    """
        config: net_ac_fe, net_actor_fm, net_critic_fm
    """

    max_action = 2.0
    max_action = 1.0

    logstd_min = -2.5
    logstd_max = 1

    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.fe = config.get('net_ac_fe', FeatureExtractor)(config, 0)

        ### 特意没有用deepcopy
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.std = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)

        self.critic = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature, 1)
        self.apply(init_weights)
        

    def forward(self, state):
        x = self.fe(state)
        mean = torch.tanh(self.mean(x)) *self.max_action
        logstd = torch.tanh(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        logstd *= 0.5
        value = self.critic(x)

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        action = dist.sample()
        logprob = dist.log_prob(action).unsqueeze(1)
        return Data(action=action, logprob=logprob, mean=mean, logstd=logstd, value=value)
    

    def evaluate(self, state, action):
        x = self.fe(state)
        mean = torch.tanh(self.mean(x)) *self.max_action
        logstd = torch.tanh(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        logstd *= 0.5
        value = self.critic(x)

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        logprob = dist.log_prob(action).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)
        return logprob, value, entropy, dist

