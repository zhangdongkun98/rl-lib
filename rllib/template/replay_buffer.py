
import numpy as np
from abc import ABC, abstractmethod

from .data import Experience


class ReplayBufferSingleAgent(ABC):
    def __init__(self, capacity, batch_size, device):
        self.capacity, self.size = capacity, 0
        self.batch_size, self.device = batch_size, device
        self._memory = np.empty(self.capacity, dtype=Experience)
        
    def push(self, experience):
        self._memory[self.size % self.capacity] = experience
        self.size += 1

    def sample(self):
        indices = np.random.randint(0, len(self), size=self.batch_size)
        batch = self._memory[indices]
        experiences = self._batch_stack(batch)
        return experiences
    
    def __len__(self):
        return np.clip(self.size, 0, self.capacity)

    def _full(self):
        return self.size >= self.capacity

    @abstractmethod
    def _batch_stack(self, batch):
        return batch


