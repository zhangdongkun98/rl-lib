
import numpy as np
from typing import List

import torch

from ..basic import Data as Experience
from .tools import stack_data


class ReplayBuffer(object):
    def __init__(self, config, capacity, batch_size, device):
        self.capacity, self.size = capacity, 0
        self.batch_size, self.device = batch_size, device
        self.memory = np.empty(self.capacity, dtype=Experience)
        self.num_visits = np.zeros(self.capacity, dtype=np.int64)
        
    def push(self, experience, **kwargs):
        self.memory[self.size % self.capacity] = experience
        self.size += 1

    def sample(self):
        batch: List[Experience] = self.get_batch(self.batch_size)
        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)
    
    def __len__(self):
        return np.clip(self.size, 0, self.capacity)

    def full(self):
        return self.size >= self.capacity


    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self), size=batch_size)

        unique, counts = np.unique(indices, return_counts=True)
        self.num_visits[unique] += counts

        batch = self.memory[indices]
        return batch

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


    def close(self):
        return

