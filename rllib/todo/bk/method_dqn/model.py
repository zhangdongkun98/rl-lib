
import os
from os.path import join
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

file_dir = os.path.dirname(__file__)


class QNet(nn.Module):
    def __init__(self, model_id, config):
        super(QNet, self).__init__()
        self.model_id = model_id
        self.num_vehicles = config.num_vehicles
        self.dim_other_obsv = config.dim_other_obsv
        self.dim_action = config.dim_action

        in_channels, out_channels = 3, 256
        self.cnn = CNN(in_channels, out_channels)
        self.fc = FC(input_dim=out_channels+self.dim_other_obsv, output_dim=self.dim_action)
    
    def forward(self, observation):
        image, other_obsv = observation[0], observation[1]
        feature = self.cnn(image)
        x = torch.cat([feature, other_obsv], dim=1)
        result = self.fc(x)
        return result
    
    def load_model(self):
        self.load_state_dict(torch.load( glob.glob(join(file_dir, 'qnet_%d*.pth' % self.model_id))[0] ))
    def save_model(self, path, iter_num):
        torch.save(self.state_dict(), join(path, 'qnet_%d_%d.pth' % (self.model_id, iter_num)))



class CNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(CNN, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64,  128, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(128, out_channels, 3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.apply(weights_init)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)     

        x = self.conv3(x)
        x = F.leaky_relu(x)
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=3, stride=1)
        # print(x.shape)
        # print('---')

        # x = self.conv4(x)
        # x = F.leaky_relu(x)

        # print(x.shape)
        x = x.view(batch_size, -1)
        # print(x.shape)
        return x
        
class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, output_dim)
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        x = torch.tanh(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear4(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)
