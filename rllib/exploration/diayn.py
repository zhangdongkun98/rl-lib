
import copy
import numpy as np
from gym import Wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal

from rllib.buffer import ReplayBuffer
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper

from rllib.sac import SAC


class DIAYN(SAC):
    lr_discriminator = 0.0003

    discriminator_freq = 10
    buffer_size = 100000
    batch_size = 128

    def __init__(self, config, writer):
        config.dim_state += 1
        super().__init__(config, writer)

        self.discriminator = config.get('net_discriminator', Discriminator)(config).to(self.device)
        self.models_to_save += [self.discriminator]

        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.discriminator_loss = nn.CrossEntropyLoss()


    def update_parameters(self):
        res = super().update_parameters()
        if self.step_update % self.discriminator_freq == 0:
            self.update_discriminator()
        return res

    def update_discriminator(self):
        if len(self.buffer) < self.start_timesteps:
            return
        
        experience = self.buffer.sample()
        state = experience.state[:,:-1]
        skill = experience.info.skill

        ### check if has problems
        loss = self.discriminator_loss(self.discriminator(state), skill)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        self.writer.add_scalar('method/loss_discriminator', loss.detach().item(), self.step_update)
        return




class Discriminator(Model):
    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.fe = config.get('net_net_discriminator_fe', DiscriminatorFeatureExtractor)(config, 0)
        self.fm = config.get('net_net_discriminator_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.num_skills)
        self.apply(init_weights)

    def forward(self, state):
        return self.fm(self.fe(state))


class DiscriminatorFeatureExtractor(object):
    def __init__(self, config, model_id):
        self.dim_feature = config.dim_state -1
    def __call__(self, x, **kwargs):
        return x




class EnvWrapper(Wrapper):
    def __init__(self, env, method: DIAYN, num_skills, mode='train'):
        super().__init__(env)
        self.method = method
        self.num_skills = num_skills
        self.mode = mode
        self.step_reset = 0

    def reset(self, **kwargs):
        self.step_reset += 1
        observation = self.env.reset(**kwargs)
        if self.mode == 'train':
            self.skill = np.random.randint(self.num_skills)
        else:
            self.skill = self.step_reset % self.num_skills
        return self.observation(observation)

    def step(self, action):
        next_state, _, done, _ = self.env.step(action)
        reward = self.calculate_reward(next_state)
        info = {'skill': torch.tensor([self.skill])}
        return self.observation(next_state), reward, done, info

    def observation(self, observation):
        return np.concatenate([observation, np.array([self.skill])])


    def calculate_reward(self, next_state):
        next_state = torch.Tensor(next_state).unsqueeze(0).to(self.method.device)
        discriminator_output = self.method.discriminator(next_state)
        discriminator_output_probability = F.softmax(discriminator_output, dim=1)

        # skill = torch.tensor([self.skill]).to(self.method.device)
        # self.method.update_discriminator(discriminator_output, skill)

        reward = np.log(discriminator_output_probability[:,self.skill].item() + 1e-8) - np.log(1 /self.num_skills)
        # print('reward: ', self.skill, discriminator_output.squeeze().cpu(), discriminator_output_probability.squeeze().cpu(), reward)
        return reward


