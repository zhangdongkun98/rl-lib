import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class MixNet(nn.Module):
    def __init__(self):
        super(MixNet, self).__init__()
        self.conf = Config()
        self.hyper_w1 = nn.Linear(self.conf.state_shape * self.conf.n_agents, self.conf.qmix_hidden_dim * self.conf.n_agents)
        self.hyper_w2 = nn.Linear(self.conf.state_shape * self.conf.n_agents, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape * self.conf.n_agents, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape * self.conf.n_agents, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))

    def forward(self, q_values, states):
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        states = states.reshape(-1, self.conf.state_shape * self.conf.n_agents)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        try:
            hidden = F.elu(torch.bmm(q_values, w1) + b1)
        except:
            import pdb; pdb.set_trace()

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(32, -1, 1)

        return q_total
