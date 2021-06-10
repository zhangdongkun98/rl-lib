
import os

import torch
import torch.nn as nn

from utils import Model, DeepSet, init_weights


class Critic(Model):
    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        self.dim_rnn_hidden = 128
        self.rnn_layer = 1
        self.lstm = nn.LSTM(
            input_size=(dim_state + dim_action),
            hidden_size=self.dim_rnn_hidden,
            num_layers=self.rnn_layer,
            bias=True,
            batch_first=True,
            dropout=0.0)
        self.fc1_v1 = nn.Sequential(
            nn.Linear(self.dim_rnn_hidden*self.rnn_layer, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.fc2_v1 = nn.Sequential(
            nn.Linear(self.dim_rnn_hidden*self.rnn_layer, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim_state*8+dim_action, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim_state*8+dim_action, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.deepset = DeepSet(dim_state, dim_output=64)

        self.apply(init_weights)
   
    def forward_v1(self, state, action):
        sa = torch.cat([state, action], 1)
        self.lstm.flatten_parameters()
        _, (x, _) = self.lstm(sa.unsqueeze(1))
        x = x.transpose(0,1).reshape(x.shape[1],-1)
        return self.fc1(x), self.fc2(x)
    
    def q1_v1(self, state, action):
        sa = torch.cat([state, action], 1)
        _, (x, _) = self.lstm(sa.unsqueeze(1))
        x = x.transpose(0,1).reshape(x.shape[1],-1)
        return self.fc1(x)
    

    def forward(self, state, action):

        self.deepset(state)


        x = torch.cat([state.view(state.shape[0],-1), action], dim=1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        x = torch.cat([state.view(state.shape[0],-1), action], dim=1)
        return self.fc1(x)



class Actor(Model):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.file_dir = os.path.dirname(__file__)

        self.dim_rnn_hidden = 128
        self.rnn_layer = 1
        self.lstm = nn.LSTM(
            input_size=(dim_state),
            hidden_size=self.dim_rnn_hidden,
            num_layers=self.rnn_layer,
            bias=True,
            batch_first=True,
            dropout=0.0)
        self.fc_v1 = nn.Sequential(
            nn.Linear(self.dim_rnn_hidden*self.rnn_layer, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_action), nn.Softmax(dim=1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dim_state*8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_action), nn.Softmax(dim=1),
        )

        self.deepset = DeepSet(dim_state, dim_output=64)

        self.apply(init_weights)
    
    def forward_v1(self, state):
        self.lstm.flatten_parameters()
        _, (x, _) = self.lstm(state.unsqueeze(1))
        x = x.transpose(0,1).reshape(x.shape[1],-1)
        return self.fc(x)
    

    def forward(self, x_fixed, x_dynamic, lengths):
        print('here')
        x_dynamic = self.deepset(x_dynamic, lengths)

        import pdb; pdb.set_trace()

        x = state.view(state.shape[0],-1)
        x = self.fc1(x)
        return x