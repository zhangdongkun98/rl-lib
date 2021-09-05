
import numpy as np
from abc import ABC, abstractmethod

import torch

from ..basic import Data as Experience
from .tools import stack_data


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size, device):
        self.capacity, self.size = capacity, 0
        self.batch_size, self.device = batch_size, device
        self._memory = np.empty(self.capacity, dtype=Experience)
        self.num_visits = np.zeros(self.capacity, dtype=np.int64)
        
    def push(self, experience):
        self._memory[self.size % self.capacity] = experience
        self.size += 1

    def sample(self):
        indices = np.random.randint(0, len(self), size=self.batch_size)

        unique, counts = np.unique(indices, return_counts=True)
        self.num_visits[unique] += counts

        batch = self._memory[indices]
        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)
    
    def __len__(self):
        return np.clip(self.size, 0, self.capacity)

    def full(self):
        return self.size >= self.capacity

    @abstractmethod
    def _batch_stack(self, batch):
        return batch



class ReplayBufferOffPolicy(ReplayBuffer):
    def _batch_stack(self, batch):
        result = stack_data(batch)

        result.update(reward=[*torch.tensor(result.reward, dtype=torch.float32).unsqueeze(1)])
        result.update(done=[*torch.tensor(result.done, dtype=torch.float32).unsqueeze(1)])
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result
