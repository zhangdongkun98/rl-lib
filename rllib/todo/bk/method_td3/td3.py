import carla_utils as cu

import numpy as np
from copy import deepcopy
import random

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=6, linewidth=65536, suppress=True)

from utils import soft_update
from utils.method_single_agent import Method
from .model import Critic, Actor
from utils.replay_buffer import ReplayBuffer


from collections import namedtuple
TmpState = namedtuple('TmpState', ('fixed', 'dynamic', 'mask'))

class TD3(Method):
    def __init__(self, device, config, path_pack):
        super(TD3, self).__init__(device, config, path_pack)

        '''param'''
        self.gamma = config.gamma
        self.tau = config.tau

        '''network'''
        self.critic = Critic(config.dim_state, config.dim_action).to(device)
        self.actor = Actor(config.dim_state, config.dim_action).to(device)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        self.models_to_load = [self.critic, self.actor, self.critic_target, self.actor_target]
        self.models_to_save = [self.critic, self.actor]
        if config.load_model is True: self._load_model()

        self.critic_optimizer= Adam(self.critic.parameters(), lr=5e-4)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)

        self.critic_loss = nn.MSELoss()

        self.memory = ReplayBuffer(config, device)

        self.epsilon_prob = config.epsilon_prob_start
        self.decay_rate = (config.epsilon_prob_end / config.epsilon_prob_start) ** (1/config.n_episode)
        self.decay_rate = config.decay_rate


    def update_policy(self):
        self.train_step += 1
        if self.train_step % 100 == 0:
            print('[update_policy] buffer size: ', len(self.memory))
        self.epsilon_prob *= self.decay_rate
        # self.epsilon_prob = np.clip(self.epsilon_prob, 0.1, 1.0)
        
        '''load data batch'''
        experience_batch = self.memory.sample()
        state = experience_batch.states
        action = experience_batch.actions
        next_state = experience_batch.next_states
        reward = experience_batch.rewards
        done = experience_batch.dones

        '''critic'''
        with torch.no_grad():
            tmp_next_state = TmpState(next_state.items[:,0,:], next_state.items[:,1:,:], next_state.indices[:,1:])
            next_action = self.actor_target(tmp_next_state)

            target_q1, target_q2 = self.critic_target(tmp_next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * (1-done) * target_q

        tmp_state = TmpState(state.items[:,0,:], state.items[:,1:,:], state.indices[:,1:])
        current_q1, current_q2 = self.critic(tmp_state, action)
    
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        a_loss = 0.0
        if self.train_step % 2 == 0:
            actor_loss = -self.critic.q1(tmp_state, self.actor(tmp_state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._update_model()

            a_loss = actor_loss.detach().item()

        if self.train_step % 100 == 0:
            from os.path import join
            output = np.hstack((
                action.data.cpu().numpy(),
                done.data.cpu().numpy(),
                reward.data.cpu().numpy(),
                current_q1.data.cpu().numpy(),
                current_q2.data.cpu().numpy(),
                target_q.data.cpu().numpy(),
            ))
            np.savetxt(join(self.path_pack.output_path, str(self.train_step)+'.txt'), output, delimiter='   ', fmt='%7f')

        # print('time: ', tick2-tick1, tick3-tick2, tick4-tick3)

        if self.train_step % 100 == 0: self._save_model()
        if self.train_step % 100 == 0: torch.cuda.empty_cache()
        return critic_loss.detach().item(), a_loss


    @torch.no_grad()
    def _select_action_train(self, state):
        if random.random() < self.epsilon_prob:
            action = random.choice(list(range(self.dim_action)))
            action_prob = cu.basic.int2onehot(torch.tensor(action, dtype=torch.float32), self.dim_action)
        else:
            items = state.items.to(self.device)
            # state.indices[torch.where(state.indices > -1)]
            # obsv = items[state.indices[torch.where(state.indices > -1)]].unsqueeze(0)
            # action_prob = self.actor(obsv).squeeze()

            # index = state.indices[torch.where(state.indices > -1)][1:]

            mask = state.indices.unsqueeze(0)[:,1:].to(self.device)
            x_fixed = items.unsqueeze(0)[:,0,:]   ### torch.Size([batch_size, dim_state])
            x_dynamic = items.unsqueeze(0)[:,1:]   ### torch.Size([batch_size, num_nearby_vehicles, dim_state])

            tmp_state = TmpState(x_fixed, x_dynamic, mask)
            action_prob = self.actor(tmp_state).squeeze()
            # softmax 

            action = torch.argmax(action_prob.detach()).cpu().item()
        return action, action_prob.cpu().data
    

    @torch.no_grad()
    def _select_action_eval(self, state):

        items = state.items.to(self.device)
        # state.indices[torch.where(state.indices > -1)]
        # obsv = items[state.indices[torch.where(state.indices > -1)]].unsqueeze(0)
        # action_prob = self.actor(obsv).squeeze()

        action_prob = self.actor(items.unsqueeze(0)).squeeze()

        q = self.critic(items.unsqueeze(0), action_prob.unsqueeze(0))
        print('state: ', items)
        print('q: ', q)
        print('action: ', action_prob)
        print()

        action = torch.argmax(action_prob.detach()).cpu().item()
        return action, action_prob.cpu().data

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

