
import os

import torch
import torch.nn as nn

from utils import Model, DeepSet, init_weights


class Critic(Model):
    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        dim_dynamic_feature = 64
        self.deepset = DeepSet(dim_state, dim_output=dim_dynamic_feature)
        self.fc1 = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.apply(init_weights)
   
    def forward(self, state, action):
        x_dynamic = self.deepset(state.dynamic, state.mask)
        x = torch.cat([state.fixed, action, x_dynamic], dim=1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        x_dynamic = self.deepset(state.dynamic, state.mask)
        x = torch.cat([state.fixed, action, x_dynamic], dim=1)
        return self.fc1(x)


class Actor(Model):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        dim_dynamic_feature = 64
        self.deepset = DeepSet(dim_state, dim_output=dim_dynamic_feature)
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_action), nn.Softmax(dim=1),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        x_dynamic = self.deepset(state.dynamic, state.mask)
        x = torch.cat([state.fixed, x_dynamic], dim=1)
        return self.fc(x)
