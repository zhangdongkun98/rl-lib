import carla_utils as cu

import os
import numpy as np
from copy import deepcopy
import random

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=6, linewidth=65536, suppress=True)

from utils import soft_update, init_weights
from utils.method_single_agent import Method, Model
from utils.replay_buffer import ReplayBuffer


class DDPG(Method):
    def __init__(self, device, config, path_pack):
        super(DDPG, self).__init__(device, config, path_pack)

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
        
        '''load data batch'''
        experience_batch = self.memory.sample()
        state = experience_batch.states.items
        action = experience_batch.actions
        next_state = experience_batch.next_states.items
        reward = experience_batch.rewards
        done = experience_batch.dones

        '''critic'''
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (self.gamma * (1-done) * target_q)
        
        current_q = self.critic(state, action)
        critic_loss = self.critic_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action_new = self.actor(state)
        actor_loss = -self.critic(state, action_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.train_step % 100 == 0:
            from os.path import join
            output = np.hstack((
                action.data.cpu().numpy(),
                done.data.cpu().numpy(),
                reward.data.cpu().numpy(),
                current_q.data.cpu().numpy(),
                target_q.data.cpu().numpy(),
                action_new.data.cpu().numpy(),
            ))
            np.savetxt(join(self.path_pack.output_path, str(self.train_step)+'.txt'), output, delimiter='   ', fmt='%7f')

        self._update_model()
        if self.train_step % 300 == 0: self._save_model()
        if self.train_step % 300 == 0: torch.cuda.empty_cache()
        return critic_loss.detach().item(), actor_loss.detach().item()


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

            action_prob = self.actor(items.unsqueeze(0)).squeeze()

            action = torch.argmax(action_prob.detach()).cpu().item()
        self.epsilon_prob *= self.decay_rate
        return action, action_prob.cpu().data
    
    @torch.no_grad()
    def _select_action_eval(_): return

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



class Critic(Model):
    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        self.dim_rnn_hidden = 128
        self.rnn_layer = 1
        self.lstm = nn.LSTM(
            input_size=dim_state,
            hidden_size=self.dim_rnn_hidden,
            num_layers=self.rnn_layer,
            bias=True,
            batch_first=True,
            dropout=0.0)
        self.fc = nn.Sequential(
            nn.Linear(self.rnn_layer*self.dim_rnn_hidden+dim_action, 400), nn.ReLU(inplace=True),
            nn.Linear(400, 300), nn.ReLU(inplace=True),
            nn.Linear(300, 1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim_state*8+dim_action, 400), nn.ReLU(inplace=True),
            nn.Linear(400, 300), nn.ReLU(inplace=True),
            nn.Linear(300, 1),
        )
        self.apply(init_weights)
   
    def forward_v1(self, state, action):
        self.lstm.flatten_parameters()
        _, (x, _) = self.lstm(state)
        x = x.transpose(0,1).reshape(x.shape[1],-1)
        x = torch.cat([x, action], dim=1)
        return self.fc(x)
    
    def forward(self, state, action):
        x = torch.cat([state.view(state.shape[0],-1), action], dim=1)
        return self.fc1(x)


class Actor(Model):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        # self.dim_rnn_hidden = 128
        # self.rnn_layer = 1
        # self.lstm = nn.LSTM(
        #     input_size=dim_state,
        #     hidden_size=self.dim_rnn_hidden,
        #     num_layers=self.rnn_layer,
        #     bias=True,
        #     batch_first=True,
        #     dropout=0.0)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.rnn_layer*self.dim_rnn_hidden, 400), nn.ReLU(inplace=True),
        #     nn.Linear(400, 300), nn.ReLU(inplace=True),
        #     nn.Linear(300, dim_action), nn.Softmax(dim=1),
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(dim_state*8, 400), nn.ReLU(inplace=True),
            nn.Linear(400, 300), nn.ReLU(inplace=True),
            nn.Linear(300, dim_action), nn.Softmax(dim=1),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        self.lstm.flatten_parameters()
        _, (x, _) = self.lstm(state)
        x = x.transpose(0,1).reshape(x.shape[1],-1)
        x = self.fc(x)
        return x
    

    def forward(self, state):
        x = state.view(state.shape[0],-1)
        x = self.fc1(x)
        return x


