
import copy
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from .buffer import RolloutBuffer
from .utils import init_weights, hard_update
from .template import MethodSingleAgent, Model
from .template.model import FeatureExtractor, FeatureMapper


class PPO(MethodSingleAgent):
    gamma = 0.99
    epsilon_clip = 0.2
    weight_value = 1.0
    weight_entropy = 0.001

    lr = 0.0003
    betas = (0.9, 0.999)

    K_epochs = 4
    K_epochs = 10
    buffer_size = 2000
    batch_size = 0

    save_model_interval = 20

    def __init__(self, config, writer):
        super(PPO, self).__init__(config, writer)

        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        self.critic_loss = nn.MSELoss()
        self._memory: RolloutBuffer = config.get('buffer', RolloutBuffer)(self.device, self.batch_size)


    def update_parameters(self):
        if len(self._memory) < self.buffer_size:
            return
        super().update_parameters()

        for _ in range(self.K_epochs):
            self.step_train += 1

            experience = self._memory.sample(self.gamma)
            old_states = experience.state
            old_actions = experience.action
            old_logprobs = experience.prob
            rewards = experience.reward
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages

            loss =  - torch.min(surr1, surr2) \
                    - self.weight_entropy*dist_entropy \
                    + self.weight_value*self.critic_loss(state_values, rewards)

            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer.step()

            self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_train)


        if self.step_update % self.save_model_interval == 0:
            self._save_model()

        hard_update(self.policy_old, self.policy)
        self._memory.clear()
        return


    @torch.no_grad()
    def select_action(self, state):
        super().select_action()
        action, logprob, _ = self.policy_old(state.to(self.device))
        self._memory.push_prob(logprob)
        return action



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
        action_probs = self.actor_no(self.actor(x))
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.unsqueeze(1), dist.log_prob(action).unsqueeze(1), action_probs

    def evaluate(self, state, action):
        x = self.fe(state)
        action_probs = self.actor_no(self.actor(x))
        state_value = self.critic(x)

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action.squeeze()).unsqueeze(1)
        dist_entropy = dist.entropy().unsqueeze(1)

        return action_logprobs, state_value, dist_entropy



class ActorCriticContinuous(Model):
    """
        config: net_ac_fe, net_actor_fm, net_critic_fm
    """

    max_action = 2.0

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

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        action = dist.sample()
        logprob = dist.log_prob(action).unsqueeze(1)
        return action, logprob, mean
    

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
        return logprob, value, entropy

