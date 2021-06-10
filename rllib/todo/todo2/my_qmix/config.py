import torch


class Config(object):
    def __init__(self):
        self.BATCH_SIZE = 32
        self.LR = 1e-4
        self.epsilon = 0.9
        self.GAMMA = 0.9
        self.TARGET_REPLACE_ITER = 40
        self.MEMORY_CAPACITY = 1000

        self.N_ACTIONS = 3
        self.N_STATES = 1156
        self.ENV_A_SHAPE = 0
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.state_shape = 1156
        self.n_agents = 8
        self.obs_shape = self.state_shape * self.n_agents
        self.qmix_hidden_dim = 64

        self.train_episodes = 1500
        self.update_target_params = 100