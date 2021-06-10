import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        """
        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )
        """
        # cnn_actor
        #
        self.action_layer = nn.Sequential(
            nn.Conv2d(2, 32, 2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(n_latent_var, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # cnn_critic
        #
        #             nn.Conv2d(3, 64, 3),
        #             nn.Tanh(),
        #             nn.Conv2d(64, 64, 3),
        #             nn.Tanh(),
        #             nn.Conv2d(64, 3, 3),
        #             nn.Tanh(),
        #             nn.Flatten(),
        #             """
        self.value_layer = nn.Sequential(
            nn.Conv2d(2, 32, 2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 2),
            nn.Tanh(),
            nn.Conv2d(32, 32, 2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(n_latent_var, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError


def restore():