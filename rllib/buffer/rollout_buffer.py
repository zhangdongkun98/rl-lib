
import numpy as np
from typing import List

import torch

from ..basic import Data as Experience
from .tools import stack_data


class RolloutBuffer(object):
    def __init__(self, config, device, batch_size=-1, use_gae=False):
        self.batch_size, self.device = batch_size, device
        self.use_gae = use_gae
        self.memory: List[Experience] = []
        self.rollout_reward = False

    def push(self, experience: Experience):
        self.memory.append(experience)


    def sample(self, gamma, gae_lambda=0.9, advantage_normalization=False):
        if self.use_gae:
            self.compute_returns_and_advantage(gamma, gae_lambda, advantage_normalization)
        else:
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


    def get_batch(self, batch_size):
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = self.memory[indices]
        return batch

    def _batch_stack(self, batch):
        """
            To be override.
        """

        result = stack_data(batch)
        result = result.cat(dim=0)
        result.reward = result.reward.unsqueeze(1).to(torch.float32)
        result.done = result.done.unsqueeze(1).to(torch.float32)
        return result


    def reward2return(self, gamma):
        if self.rollout_reward:
            return
        
        self.rollout_reward = True

        returns = torch.zeros(len(self), dtype=torch.float32)
        discounted_reward = 0
        for idx in reversed(range(len(self))):
            e = self.memory[idx]
            if e.done:
                discounted_reward = 0
            discounted_reward = e.reward + gamma * discounted_reward
            returns[idx] = discounted_reward
        
        returns = returns[..., None,None]
        for e, r in zip(self.memory, returns):
            e.update(returns=r)
        
        self.memory = np.array(self.memory, dtype=Experience)
        return


    def compute_returns_and_advantage(self, gamma, gae_lambda, advantage_normalization):
        if self.rollout_reward:
            return
        
        self.rollout_reward = True
        last_gae_lam = 0

        assert self.memory[-1].done == True

        advantages = torch.zeros(len(self), dtype=torch.float32)
        returns = torch.zeros(len(self), dtype=torch.float32)
        for idx in reversed(range(len(self))):
            e = self.memory[idx]
            e_next = self.memory[min(idx +1, len(self)-1)]

            next_non_terminal = 1.0 - e_next.done
            next_value = e_next.action_data.value.item()

            delta = e.reward + gamma * next_value * next_non_terminal - e.action_data.value.item()
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

            advantages[idx] = last_gae_lam
            returns[idx] = advantages[idx] + e.action_data.value.item()

        if advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + np.finfo(np.float32).eps)

        advantages = advantages[..., None,None]
        returns = returns[..., None,None]
        for e, r, ad in zip(self.memory, returns, advantages):
            e.update(returns=r, advantage=ad)
        self.memory = np.array(self.memory, dtype=Experience)
        return


