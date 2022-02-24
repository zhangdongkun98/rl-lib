
import numpy as np
from typing import List

import torch

from ..basic import Data as Experience
from .tools import stack_data


class RolloutBuffer(object):
    def __init__(self, config, device, batch_size=-1):
        self.batch_size, self.device = batch_size, device
        self.memory: List[Experience] = []
        self.rollout_reward = False

    def push(self, experience: Experience):
        self.memory.append(experience)


    def sample(self, gamma):
        self.reward2return(gamma)
        batch_size = len(self) if self.batch_size <= 0 else self.batch_size
        batch = self.get_batch(batch_size)
        experiences: Experience = self._batch_stack(batch)
        return experiences.to(self.device)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        del self.memory
        self.memory = []
        self.rollout_reward = False


    def reward2return(self, gamma):
        if not self.rollout_reward:
            self.rollout_reward = True

            rewards = []
            discounted_reward = 0
            for e in reversed(self.memory):
                if e.done:
                    discounted_reward = 0
                discounted_reward = e.reward + gamma * discounted_reward
                rewards.insert(0, discounted_reward)
            
            for e, reward in zip(self.memory, rewards):
                e.update(reward=reward)
            
            self.memory = np.array(self.memory, dtype=Experience)
        return

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
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
