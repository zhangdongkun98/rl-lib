import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cpu")


class ActorCritic(nn.Module):
    def __init__(self, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
            nn.Linear(n_latent_var, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(n_latent_var, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_list = []
        for tensor_action in list(action):
            action_list.append(tensor_action.item())
        return action_list

class Policy():
    def __init__(self):
        self.policy = ActorCritic(3, 288).to(device)
        self.restore()

    def choose_action(self, state):
        return self.policy.act(state)

    def restore(self):
        self.policy.load_state_dict(torch.load('./net_30000_5x5_grid_vy.pth'))