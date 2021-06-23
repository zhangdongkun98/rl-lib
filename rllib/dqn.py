

import copy
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from .utils import init_weights, soft_update
from .template import MethodSingleAgent, Model, ReplayBufferSingleAgent, Experience


class DQN(MethodSingleAgent):
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
        super(DQN, self).__init__(config, writer)

        self.policy = config.get('net_ac', ActorCritic)(config).to(self.device)
        self.policy_target = copy.deepcopy(self.policy)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self._replay_buffer = config.get('buffer', ReplayBuffer)(self.buffer_size, self.batch_size, config.device)


    def update_policy(self):
        if len(self._replay_buffer) < self.start_timesteps:
            return
        super().update_policy()

        '''load data batch'''
        experience = self._replay_buffer.sample()
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

        self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_update)
        self.writer.add_scalar('index/epsilon_prob', self.epsilon_prob, self.step_update)
        if self.step_update % self.save_model_interval == 0: self._save_model()
        return


    @torch.no_grad()
    def select_action(self, state):
        super().select_action()

        if random.random() < self.epsilon_prob:
        # if False:
            action = torch.tensor(random.choice(range(self.dim_action))).reshape(1,-1)
        else:
            action_value = self.policy(state.to(self.device))
            action = torch.argmax(action_value, dim=1).cpu().reshape(1,-1)
        return action

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.policy_target, self.policy, self.tau)



class ReplayBuffer(ReplayBufferSingleAgent):
    def _batch_stack(self, batch):
        state, action, next_state, reward, done = [], [], [], [], []
        for e in batch:
            state.append(e.state)
            action.append(e.action)
            next_state.append(e.next_state)
            reward.append(e.reward)
            done.append(e.done)

        state = torch.cat(state, dim=0)
        action = torch.cat(action, dim=0)
        next_state = torch.cat(next_state, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        experience = Experience(
            state=state,
            next_state=next_state,
            action=action, reward=reward, done=done).to(self.device)
        return experience



class ActorCritic(Model):
    def __init__(self, config):
        super(ActorCritic, self).__init__(config, model_id=0)

        self.fc = nn.Sequential(
            nn.Linear(config.dim_state, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, config.dim_action),
        )
        self.apply(init_weights)


    def forward(self, state):
        return self.fc(state)
    


