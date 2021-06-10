import carla_utils as cu

import numpy as np
from copy import deepcopy
import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=6, linewidth=65536, suppress=True)

from utils import parse_observation, parse_observations
from utils.models import soft_update
from .model import QNet
from .memory import ReplayMemory, Experience

file_dir = os.path.dirname(__file__)


class DQN(object):
    def __init__(self, device, config, path_pack):
        self.device = device
        self.path_pack = path_pack

        '''param'''
        self.dim_action = config.dim_action
        self.batch_size = config.batch_size

        self.gamma = config.gamma
        self.tau = config.tau

        '''network'''
        self.q_net = QNet(0, config).to(device)
        self.q_net_target = deepcopy(self.q_net)
        if config.load_model == True: self._load_model()

        self.optimizer = Adam(self.q_net.parameters(), lr=1e-4, weight_decay=5e-4)

        self.critic_loss = nn.MSELoss()

        self.memory = ReplayMemory(config.capacity)

        self.steps_done = 0
        self.epsilon_prob = config.epsilon_prob_start * float(not config.eval)
        self.decay_rate = (config.epsilon_prob_end / config.epsilon_prob_start) ** (1/config.n_episode)
        # self.decay_rate = config.decay_rate

        self.eval = config.eval

    
    def update_policy(self):
        if self.memory.full() is False:
            print('[update_policy] memory buffer too short')
            print('[update_policy] buffer size: ', len(self.memory))
            return None
        
        self.steps_done += 1
        print('[update_policy] step: ', self.steps_done)
        print("[update_policy] learning !!!")
        print('[update_policy] buffer size: ', len(self.memory))
        print('[update_policy] epsilon_prob: ', self.epsilon_prob)
        self.epsilon_prob *= self.decay_rate

        loss = 0.0
        vehicle_index = 0

        '''load data batch'''
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.stack(batch.states).to(self.device)
        action_batch = torch.stack(batch.actions).to(self.device)
        next_state_batch = torch.stack(batch.next_states).to(self.device)
        reward_batch = torch.stack(batch.rewards).to(self.device)
        done_batch = torch.stack(batch.dones).to(self.device)

        '''q_net'''
        self.optimizer.zero_grad()
        current_Q = self.q_net(parse_observations(state_batch)[0])
        current_Q = torch.gather(current_Q, dim=1, index=cu.basic.onehot2int(action_batch)).squeeze()

        target_Qp = self.q_net_target(parse_observations(next_state_batch)[0]).max(dim=1)[0]
        target_Q = (target_Qp * self.gamma * (1-done_batch[:,vehicle_index])).detach() + reward_batch[:,vehicle_index]

        loss_Q = self.critic_loss(current_Q, target_Q)
        loss_Q.backward()
        self.optimizer.step()

        print(np.hstack((
            cu.basic.onehot2int(action_batch).data.cpu().numpy(),
            reward_batch[:,vehicle_index].unsqueeze(1).data.cpu().numpy(),
            current_Q.unsqueeze(-1).data.cpu().numpy(),
            target_Q.unsqueeze(-1).data.cpu().numpy(),
        )))

        '''loss'''
        loss += loss_Q.detach().item()
        
        print('\n\n\n')

        self._update_model()
        if self.steps_done % 100 == 0:
            self._save_model()

        return loss

        

    def select_action(self, observations):
        """
            Used for env.
        
        Args:
            observations.shape[0]: equals to 1
        
        Returns:
            
        """

        actions = np.zeros(1, dtype=np.int64)
        actions_value = self.get_actions_value( torch.from_numpy(observations).to(self.device) )
        if random.random() > self.epsilon_prob:
            action = torch.argmax(actions_value, dim=1)
        else:
            action = random.choice(range(self.dim_action))
        if self.eval == True:
            raise NotImplementedError
        actions[0] = int(action)
        return actions
    

    def get_actions_value(self, observation, use_target=False):
        """
        running in 'self.device'.
        
        Args:
            observation: torch.Size([batch_size, dim_observation])
            use_target: whether use target network
        
        Returns:
            the probability of actions
        """      

        _, image, other_obsv = parse_observation(observation)
        net = self.q_net if use_target == False else self.q_net_target
        actions_value = net((image, other_obsv))
        return actions_value


    def _update_model(self):
        print('[update_policy] soft update')
        soft_update(self.q_net_target, self.q_net, self.tau)

    def _save_model(self):
        print("[update_policy] save model")
        self.q_net.save_model(self.path_pack.save_model_path, self.steps_done)
    
    def _load_model(self):
        print('[update_policy] load model')
        self.q_net.load_model()
        self.q_net_target.load_model()
        return
