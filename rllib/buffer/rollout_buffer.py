
import numpy as np
from abc import ABC, abstractmethod
from typing import List

import torch

from ..basic import Data as Experience


class RolloutBuffer(object):
    def __init__(self, device, batch_size=-1):
        self.batch_size, self.device = batch_size, device
        self._states, self._actions = [], []
        self._rewards, self._dones = [], []
        self._probs = []

        self._memory: List[Experience] = []
        self._prob_cache = None
        self.rollout_reward = False

    def push(self, experience: Experience):
        experience.update(prob=self._prob_cache)
        self._memory.append(experience)

    def push_prob(self, prob):
        self._prob_cache = prob


    def sample(self, gamma):
        if not self.rollout_reward:
            rewards = []
            discounted_reward = 0
            for e in reversed(self._memory):
                if e.done:
                    discounted_reward = 0
                discounted_reward = e.reward + gamma * discounted_reward
                rewards.insert(0, discounted_reward)
            
            for e, reward in zip(self._memory, rewards):
                e.update(reward=reward)
            
            self._memory = np.array(self._memory, dtype=Experience)

        batch_size = len(self) if self.batch_size <= 0 else self.batch_size
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = self._memory[indices]

        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)


    def __len__(self):
        return len(self._memory)

    def clear(self):
        del self._memory
        self._memory = []
        self._prob_cache = None
        self.rollout_reward = False


    @abstractmethod
    def _batch_stack(self, batch):
        return batch



class RolloutBufferOnPolicy(RolloutBuffer):
    def _batch_stack(self, batch):
        state, action, prob, reward, done = [], [], [], [], []
        for e in batch:
            state.append(e.state)
            action.append(e.action)
            prob.append(e.prob)
            reward.append(e.reward)
            done.append(e.done)

        state = torch.cat(state, dim=0)
        action = torch.cat(action, dim=0)
        prob = torch.cat(prob, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        experience = Experience(
            state=state,
            prob=prob,
            action=action, reward=reward, done=done)
        return experience

