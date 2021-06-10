
import numpy as np
from collections import namedtuple

Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards', 'dones'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity, self.size = capacity, 0
        self.memory = []
        self.memory = np.empty(capacity, dtype=object)

    def push(self, *args):
        self.memory[self.size % self.capacity] = Experience(*args)
        self.size += 1

    def sample(self, batch_size):
        mask = np.random.randint(len(self), size=batch_size)
        return self.memory[mask]

    def __len__(self):
        return np.clip(self.size, None, self.capacity)
    
    def full(self):
        return self.size >= self.capacity
