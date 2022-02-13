

import copy
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from .buffer import ReplayBuffer
from .utils import init_weights, soft_update
from .template import MethodSingleAgent, Model
from .template.model import FeatureExtractor, FeatureMapper


class DoubleDQN(MethodSingleAgent):
    gamma = 0.99
    
    lr = 0.0003

    tau = 0.005

    buffer_size = 1000000
    batch_size = 256

    epsilon_prob = 0.8
    decay_rate = 0.99999

    start_timesteps = 30000

    save_model_interval = 200

    def __init__(self, config, writer):
        super(DoubleDQN, self).__init__(config, writer)

        self.policy = config.get('net_ac', ActorCritic)(config).to(self.device)
        self.policy_target = copy.deepcopy(self.policy)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.buffer = config.get('buffer', ReplayBuffer)(config, self.buffer_size, self.batch_size, config.device)


    def update_parameters(self):
        if len(self.buffer) < self.start_timesteps:
            return
        self.update_parameters_start()

        '''load data batch'''
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        self.epsilon_prob *= self.decay_rate

        with torch.no_grad():
            target_q = self.policy_target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + self.gamma * (1-done) * target_q

        current_q = self.policy(state)
        current_q = torch.gather(current_q, dim=1, index=action)

        loss = self.loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('method/loss', loss.detach().item(), self.step_update)
        self.writer.add_scalar('method/epsilon_prob', self.epsilon_prob, self.step_update)
        if self.step_update % self.save_model_interval == 0: self._save_model()
        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()

        if random.random() < self.epsilon_prob:
            action = torch.tensor(random.choice(range(self.dim_action))).reshape(1,-1)
        else:
            action_value = self.policy(state.to(self.device))
            action = torch.argmax(action_value, dim=1).cpu().reshape(1,-1)
        return action

    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.policy_target, self.policy, self.tau)



class ActorCritic(Model):
    def __init__(self, config):
        super(ActorCritic, self).__init__(config, model_id=0)

        self.fe = config.get('net_ac_fe', FeatureExtractor)(config, 0)
        self.fm = config.get('net_ac_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.apply(init_weights)


    def forward(self, state):
        x = self.fe(state)
        return self.fm(x)

