
import numpy as np
import copy
import random

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=6, linewidth=65536, suppress=True)

from .utils import init_weights, soft_update
from .template import MethodSingleAgent, Model, ReplayBufferSingleAgent, Experience


class TD3(MethodSingleAgent):
    gamma = 0.99
    tau = 0.005
    buffer_size = 1000000
    batch_size = 256
    policy_freq = 4

    explore_noise = 0.1
    policy_noise = 0.4
    noise_clip = 0.6

    start_timesteps = 30000

    def __init__(self, config, writer):
        super(TD3, self).__init__(config, writer)

        self.critic = config.net_critic(config).to(self.device)
        self.actor = config.net_actor(config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=5e-4)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_loss = nn.MSELoss()

        self._replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size, config.device)


    def update_policy(self):
        if len(self._replay_buffer) < self.start_timesteps:
            return
        super().update_policy()

        '''load data batch'''
        experience = self._replay_buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

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
        if self.step_update % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._update_model()

            self.writer.add_scalar('loss/a_loss', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/c_loss', critic_loss.detach().item(), self.step_update)
        
        if self.step_update % 200 == 0: self._save_model()
        return

    @torch.no_grad()
    def select_action(self, state):
        super().select_action()

        if self.step_select < self.start_timesteps:
            action_normal = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            noise = torch.normal(0, self.explore_noise, size=(1,self.dim_action)).to(self.device)
            action_normal = self.actor(state.to(self.device))
            # action_normal = (torch.tensor(action_normal) + noise).clamp(-1,1)
            action_normal = (action_normal.clone().detach() + noise).clamp(-1,1)
        return action_normal

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



class ReplayBuffer(ReplayBufferSingleAgent):
    def _batch_stack(self, batch):
        state, action, next_state, reward, done = [], [], [], [], []
        for e in batch:
            state.append(e.state)
            action.append(e.action)
            next_state.append(e.next_state)
            reward.append(e.reward)
            done.append(e.done)

        state = torch.cat(state, dim=0)
        action = torch.cat(action, dim=0)
        next_state = torch.cat(next_state, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # print('\n\n\n\n\n-------------------------------------------TD3: ReplayBuffer')
        # print('-------------------------------------------TD3: ReplayBuffer')

        experience = Experience(
            state=state,
            next_state=next_state,
            action=action, reward=reward, done=done).to(self.device)

        # import pdb; pdb.set_trace()
        return experience



class Actor(Model):
    def __init__(self, config):
        super(Actor, self).__init__(config, model_id=0)

        self.fc = nn.Sequential(
            nn.Linear(config.dim_state, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, config.dim_action), nn.Tanh(),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        return self.fc(state)


class Critic(Model):
    def __init__(self, config):
        super(Critic, self).__init__(config, model_id=0)

        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_state+config.dim_action, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.fc2 = copy.deepcopy(self.fc1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x)

