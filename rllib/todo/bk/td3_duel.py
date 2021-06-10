import carla_utils as cu

import numpy as np
from copy import deepcopy
import random

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=6, linewidth=65536, suppress=True)

from utils import soft_update, init_weights, to_device, DeepSet, Model
from utils.method_single_agent import Method
from utils.replay_buffer import ReplayBufferSingleAgent


class TD3(Method):
    def __init__(self, config, device, path_pack):
        super(TD3, self).__init__(config, device, path_pack)

        '''param'''
        self.policy_noise, self.noise_clip = config.policy_noise, config.noise_clip

        '''network'''
        self.critic = Critic(config, self.dim_state, self.dim_action).to(device)
        self.actor = Actor(config, self.dim_state, self.dim_action).to(device)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        self.models_to_load = [self.critic, self.critic_target, self.actor, self.actor_target]
        # self.models_to_load = [self.actor, self.actor_target]  ## !warning
        self.models_to_save = [self.critic, self.actor]
        if config.load_model: self._load_model()

        self.critic_optimizer= Adam(self.critic.parameters(), lr=5e-4)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_loss = nn.MSELoss()

        self.replay_buffer = ReplayBufferSingleAgent(config, device)


    def update_policy(self):
        self.train_step += 1
        if self.train_step % 100 == 0:
            print('[update_policy] buffer size: ', len(self.replay_buffer))
        
        '''load data batch'''
        experience_batch = self.replay_buffer.sample()
        state = experience_batch.states
        action = experience_batch.actions
        next_state = experience_batch.next_states
        reward = experience_batch.rewards
        done = experience_batch.dones

        '''critic'''
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1,1)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        a_loss = 0.0
        if self.train_step % 4 == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._update_model()

            a_loss = actor_loss.detach().item()
        
        if self.train_step % 300 == 0: self._save_model()
        if self.train_step % 300 == 0: torch.cuda.empty_cache()
        return critic_loss.detach().item(), a_loss


    @torch.no_grad()
    def _select_action_train(self, state):
        self.total_timesteps += 1
        if self.total_timesteps < self.start_timesteps:
            action_normal = torch.tensor([random.uniform(-1,1)])
        else:
            std = torch.tensor([0.1])
            noise = torch.normal(0, std)
            action_normal = self.actor(to_device(state, self.device)).item()
            action_normal = (torch.tensor(action_normal) + noise).clamp(-1,1)
        return action_normal
    
    @torch.no_grad()
    def _select_action_eval(self, state):
        action_normal = self.actor(to_device(state, self.device))

        # q = self.critic(tmp_state, action_normal)
        # print('state: ', items)
        # print('q: ', q)
        # print('action: ', action_normal)
        # print()
        return action_normal.cpu().data

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



class Critic(Model):
    def __init__(self, config, dim_state, dim_action):
        super(Critic, self).__init__(config, 0)

        dim_dynamic_feature = 64
        self.deepset = DeepSet(2*dim_state + dim_action, dim_dynamic_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.fc2 = deepcopy(self.fc1)
        self.apply(init_weights)
   
    def forward(self, state, action):
        sa_fixed = torch.cat([state.fixed, action], dim=1)
        x_dynamic = torch.cat([state.dynamic, sa_fixed.unsqueeze(1).repeat(1,state.dynamic.shape[1],1)], dim=2)
        x_dynamic = self.deepset(x_dynamic, state.mask)
        x = torch.cat([sa_fixed, x_dynamic], dim=1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        sa_fixed = torch.cat([state.fixed, action], dim=1)
        x_dynamic = torch.cat([state.dynamic, sa_fixed.unsqueeze(1).repeat(1,state.dynamic.shape[1],1)], dim=2)
        x_dynamic = self.deepset(x_dynamic, state.mask)
        x = torch.cat([sa_fixed, x_dynamic], dim=1)
        return self.fc1(x)


class Actor(Model):
    def __init__(self, config, dim_state, dim_action):
        super(Actor, self).__init__(config, 0)

        dim_dynamic_feature = 64
        self.deepset = DeepSet(dim_state, dim_dynamic_feature)
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_action), nn.Tanh(),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        x_dynamic = self.deepset(state.dynamic, state.mask)
        x = torch.cat([state.fixed, x_dynamic], dim=1)
        return self.fc(x)

