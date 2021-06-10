import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

config = Config()
input_size = config.N_STATES
output_size = config.N_ACTIONS


class SubNet(nn.Module):
    def __init__(self, ):
        super(SubNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(256, 50)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, output_size)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

