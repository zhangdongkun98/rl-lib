import torch
import pickle
import random
import numpy as np
from os.path import join

import torch.nn as nn
import torch.optim as optim

from replaybuffer import ReplayMemory, huber

steps_done = 0
running_reward = None
gamma, batch_size = 0.9, 32


class Network(nn.Module):
    def __init__(self, len_state, num_quant, num_actions):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = num_actions

        self.layer1 = nn.Linear(len_state, 256)
        self.layer1.weight.data.normal_(0.5, 0.1)
        self.layer2 = nn.Linear(256, 256)
        self.layer2.weight.data.normal_(0.5, 0.1)
        self.layer3 = nn.Linear(256, num_actions * num_quant)
        self.layer3.weight.data.normal_(0.5, 0.1)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x.view(-1, self.num_actions, self.num_quant)


class QR_DQN(object):
    def __init__(self):
        eps_start, eps_end, eps_dec = 0.9, 0.1, 500
        self.eps = eps_start
        self.memory = ReplayMemory(10000)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.Z = Network(len_state=1156, num_quant=30, num_actions=3).to(self.device)
        self.Ztgt = Network(len_state=1156, num_quant=30, num_actions=3).to(self.device)

        self.optimizer = optim.Adam(self.Z.parameters(), 1e-4)
        self.tau = torch.Tensor((2 * np.arange(self.Z.num_quant) + 1) / (2.0 * self.Z.num_quant)).view(1, -1).to(self.device)
        self.step_done = 0
        self.memory_counter = 0
        self.loss = 0.

    def choose_action(self, state):
        if self.eps > 0.05:
            self.eps -= 0.00001
        self.step_done += 1
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        state = state.to(self.device)
        action = torch.randint(0, 3, (1,))
        if random.random() > self.eps:
            action = self.Z(state).mean(2).max(1)[1]
        return int(action)

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states, next_states = states.to(self.device), next_states.to(self.device)
        rewards, dones = rewards.to(self.device), dones.to(self.device)

        theta = self.Z(states)[np.arange(batch_size), actions]

        Znext = self.Ztgt(next_states).detach()
        Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]

        Ttheta = rewards + gamma * (1 - dones) * Znext_max

        diff = Ttheta.t().unsqueeze(-1) - theta
        self.loss = huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        self.loss = self.loss.mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.step_done % 100 == 0:
            self.Ztgt.load_state_dict(self.Z.state_dict())
        # print(self.optimizer.state_dict()['param_groups'][0]['lr'])

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, float(done))
        self.memory_counter += 1

    def restore(self, path, eps):
        self.Z.load_state_dict(torch.load(path))
        self.Ztgt.load_state_dict(torch.load(path))
        self.eps = eps

    def save(self, num, save_path='.'):
        torch.save(self.Z.state_dict(), join(save_path, 'multi_iql_net' + str(num) + '.pth'))
