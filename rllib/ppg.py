
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

from rllib.ppo_simple import ActorCriticDiscrete, ActorCriticContinuous


class PPG(MethodSingleAgent):
    """
        config: net_ac, buffer
    """

    gamma = 0.99
    epsilon_clip = 0.2
    weight_entropy = 0.001
    weight_clone = 1.0

    lr = 0.0003
    lr_critic = 0.0003
    betas = (0.9, 0.999)

    # num_episodes = 20
    buffer_size = 2000
    batch_size = 32
    sample_reuse = 8
    sample_reuse_auxiliary = 6

    auxiliary_freq = 32

    def __init__(self, config, writer):
        super().__init__(config, writer)
        self.step_train_auxiliary = -1

        ### param
        # self.buffer_size = config.time_tolerance * self.num_episodes
        self.num_iters = int(self.buffer_size / self.batch_size) * self.sample_reuse
        # self.num_iters_auxiliary = int(self.buffer_size / self.batch_size) * self.sample_reuse_auxiliary
        self.num_iters_auxiliary = 8

        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.critic: Critic = Critic(config).to(self.device)
        self.models_to_save = [self.policy, self.critic]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_critic = Adam(self.critic.parameters(), lr=self.lr_critic, betas=self.betas)
        self.critic_loss = nn.MSELoss()

        self.buffer: RolloutBuffer = config.get('buffer', RolloutBuffer)(config, self.device, self.batch_size)


    def update_parameters(self):
        if len(self.buffer) < self.buffer_size:
            return
        self.update_parameters_start()
        print(prefix(self) + 'update step: ', self.step_update)

        for _ in range(self.num_iters):
            self.step_train += 1

            experience = self.buffer.sample(self.gamma)
            state = experience.state
            action = experience.action
            reward = experience.reward
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)

            with torch.no_grad():
                logprob_old, value_old, _, _ = self.policy_old.evaluate(state, action)
            logprob, _, entropy, _ = self.policy.evaluate(state, action)
            value = self.critic(state)

            ratio = torch.exp(logprob - logprob_old)
            advantage = reward - value.detach()   ### ! warning; replace with value_old ?

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantage

            loss_surr = -torch.min(surr1, surr2).mean()
            loss_entropy = -self.weight_entropy* entropy.mean()
            loss = loss_surr + loss_entropy
            loss_value = self.critic_loss(value, reward)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.optimizer_critic.zero_grad()
            loss_value.backward()
            self.optimizer_critic.step()

            self.writer.add_scalar('method/loss', loss.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_surr', loss_surr.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_entropy', loss_entropy.detach().item(), self.step_train)
            self.writer.add_scalar('method/loss_value', loss_value.detach().item(), self.step_train)

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

            experience = self.buffer.sample(self.gamma)
            state = experience.state
            action = experience.action
            reward = experience.reward
            reward = (reward - reward.mean()) / (reward.std() + 1e-5)

            with torch.no_grad():
                _, _, dist_old = self.policy_old(state)
            _, value, _, dist = self.policy.evaluate(state, action)

            loss_auxiliary = self.critic_loss(value, reward)
            loss_kl_divergence = kl_divergence(dist_old, dist).mean()
            loss_joint = loss_auxiliary + self.weight_clone* loss_kl_divergence
            loss_value_auxiliary = self.critic_loss(self.critic(state), reward)

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
        action, _, _ = self.policy_old(state.to(self.device))
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
    
