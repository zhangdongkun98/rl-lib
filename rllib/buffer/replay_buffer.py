
import numpy as np

import torch

from ..basic import Data as Experience
from .tools import stack_data


class ReplayBuffer(object):
    def __init__(self, capacity, batch_size, device):
        self.capacity, self.size = capacity, 0
        self.batch_size, self.device = batch_size, device
        self.memory = np.empty(self.capacity, dtype=Experience)
        self.num_visits = np.zeros(self.capacity, dtype=np.int64)
        
    def push(self, experience):
        self.memory[self.size % self.capacity] = experience
        self.size += 1

    def sample(self):
        indices = np.random.randint(0, len(self), size=self.batch_size)

        unique, counts = np.unique(indices, return_counts=True)
        self.num_visits[unique] += counts

        batch = self.memory[indices]
        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)
    
    def __len__(self):
        return np.clip(self.size, 0, self.capacity)

    def full(self):
        return self.size >= self.capacity


    def _batch_stack(self, batch):
        """
            To be override.
        """
        
        result = stack_data(batch)

        result.update(reward=[*torch.tensor(result.reward, dtype=torch.float32).unsqueeze(1)])
        result.update(done=[*torch.tensor(result.done, dtype=torch.float32).unsqueeze(1)])
        result = result.cat(dim=0)
        result.reward.unsqueeze_(1)
        result.done.unsqueeze_(1)
        return result




class ReplayBuffer_v0(ReplayBuffer):
    def __init__(self, capacity, batch_size, device):
        super().__init__(capacity, batch_size, device)

        self.sc_memory = np.empty(self.capacity, dtype=Experience)
        self.sc_size = 0
        self.sc_num_visits = np.zeros(self.capacity, dtype=np.int64)


    def push(self, experience, flag):
        super().push(experience)
        if flag:
            self.sc_memory[self.sc_size % self.capacity] = experience
            self.sc_size += 1
        return

    def sample(self):
        sc_batch_size = 1

        indices = np.random.randint(0, len(self), size=self.batch_size -sc_batch_size)
        unique, counts = np.unique(indices, return_counts=True)
        self.num_visits[unique] += counts
        batch = self.memory[indices]

        sc_indices = np.random.randint(0, np.clip(self.sc_size, 0, self.capacity), size=sc_batch_size)
        sc_unique, sc_counts = np.unique(sc_indices, return_counts=True)
        self.sc_num_visits[sc_unique] += sc_counts
        sc_batch = self.sc_memory[sc_indices]

        batch = np.concatenate([batch, sc_batch])
        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)

