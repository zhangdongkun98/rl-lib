
import copy
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal, kl_divergence

from rllib.basic import prefix, Data
from rllib.buffer import RolloutBuffer
from rllib.utils import init_weights, hard_update
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper

from rllib.ppo import ActorCriticDiscrete, ActorCriticContinuous

class PPG(MethodSingleAgent):
    def __init__(self, config, writer):
        super().__init__(config, writer)
        self.step_train_auxiliary = -1

        ### param
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.9)

        self.batch_size = config.get('batch_size', 32)
        self.buffer_size = config.get('buffer_size', 2000)
        self.sample_reuse = config.get('sample_reuse', 8)
        self.num_iters = int(self.buffer_size / self.batch_size) * self.sample_reuse
        self.num_iters_auxiliary = 8
        self.auxiliary_freq = 32

        self.lr = config.get('lr', 0.0003)
        self.lr_critic = config.get('lr_critic', 0.0003)
        self.betas = config.get('betas', (0.9, 0.999))
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        self.epsilon_clip = config.get('epsilon_clip', 0.2)
        self.weight_value = config.get('weight_value', 0.5)
        self.weight_entropy = config.get('weight_entropy', 0.001)
        self.weight_clone = config.get('weight_clone', 0.5)

        ### model
        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.critic: Critic = Critic(config).to(self.device)
        self.models_to_save = [self.policy, self.critic]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=self.lr_critic, betas=self.betas)
        self.critic_loss = nn.MSELoss()

        self.buffer: RolloutBuffer = config.get('buffer', RolloutBuffer)(config, self.device, self.batch_size, use_gae=True)


    def update_parameters(self):
        if len(self.buffer) < self.buffer_size:
            return
        self.update_parameters_start()
        print(prefix(self) + 'update step: ', self.step_update)

        for _ in range(self.num_iters):
            self.step_train += 1

            experience = self.buffer.sample(self.gamma, self.gae_lambda, advantage_normalization=True)
            state = experience.state
            action = experience.action_data.action
            logprob_old = experience.action_data.logprob
            value_old = experience.action_data.value

            returns = experience.returns
            advantage = experience.advantage

            ### no need
            # with torch.no_grad():
            #     logprob_old, value_old, _, _ = self.policy_old.evaluate(state, action)

            logprob, _, entropy, _ = self.policy.evaluate(state, action)
            value = self.critic(state)

            ratio = torch.exp(logprob - logprob_old)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage
            loss_surr = -torch.min(surr1, surr2).mean()
            loss_entropy = -self.weight_entropy* entropy.mean()

            loss_policy = loss_surr + loss_entropy
            loss_value = self.weight_value* self.critic_loss(value, returns)

            self.optimizer.zero_grad()
            loss_policy.backward()
            self.optimizer.step()

            self.optimizer_critic.zero_grad()
            loss_value.backward()
            self.optimizer_critic.step()

            self.writer.add_scalar('method/loss_policy', loss_policy.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_surr', loss_surr.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_entropy', loss_entropy.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_value', loss_value.detach().item(), self.step_train)
            self.writer.add_scalar('method/ratio_offline', (ratio.detach() -1).abs().mean().detach().item(), self.step_train)

            self.update_callback(locals())

        hard_update(self.policy_old, self.policy)

        if self.step_update % self.auxiliary_freq == 0:
            self.update_parameters_auxiliary()

        self._save_model()

        self.buffer.clear()
        return


    def update_parameters_auxiliary(self):
        print(prefix(self) + 'update step auxiliary: ', self.step_update)

        for _ in range(self.num_iters_auxiliary):
            self.step_train_auxiliary += 1

            experience = self.buffer.sample(self.gamma, self.gae_lambda, advantage_normalization=True)
            state = experience.state
            action = experience.action_data.action
            returns = experience.returns

            with torch.no_grad():
                _, _, _, dist_old = self.policy_old.evaluate(state, action)
            _, value, _, dist = self.policy.evaluate(state, action)

            loss_auxiliary = self.critic_loss(value, returns)
            loss_kl_divergence = kl_divergence(dist_old, dist).mean()
            loss_joint = loss_auxiliary + self.weight_clone* loss_kl_divergence
            loss_value_auxiliary = self.critic_loss(self.critic(state), returns)

            self.optimizer.zero_grad()
            loss_joint.backward()
            self.optimizer.step()

            self.optimizer_critic.zero_grad()
            loss_value_auxiliary.backward()
            self.optimizer_critic.step()

            self.writer.add_scalar('method/loss_joint', loss_joint.detach().item(), self.step_train_auxiliary)
            self.writer.add_scalar('method/loss_kl_divergence', loss_kl_divergence.detach().item(), self.step_train_auxiliary)
            self.writer.add_scalar('method/loss_value_auxiliary', loss_value_auxiliary.detach().item(), self.step_train_auxiliary)


        return



    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()
        action = self.policy_old(state.to(self.device))
        return action.cpu()






class Critic(Model):
    """
        config: net_ac_fe, net_critic_fm
    """

    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.fe = config.get('net_ac_fe', FeatureExtractor)(config, 0)
        self.critic = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature, 1)
        self.apply(init_weights)
        

    def forward(self, state):
        x = self.fe(state)
        value = self.critic(x)
        return value
    

