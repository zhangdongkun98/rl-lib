
import numpy as np
from abc import ABC, abstractmethod

import torch

from ..template import Experience



class RolloutBuffer(ABC):
    def __init__(self, device, batch_size=-1):
        self.batch_size, self.device = batch_size, device
        self._states, self._actions = [], []
        self._rewards, self._dones = [], []
        self._probs = []
        self.rollout_reward = False

    def push(self, experience: Experience):
        self._states.append(experience.state)
        self._actions.append(experience.action)
        self._rewards.append(experience.reward)
        self._dones.append(experience.done)

    def push_prob(self, prob):
        self._probs.append(prob)


    def sample(self, gamma):
        if not self.rollout_reward:
            rewards = []
            discounted_reward = 0
            for reward, done in zip(reversed(self._rewards), reversed(self._dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + gamma * discounted_reward
                rewards.insert(0, discounted_reward)
            self._rewards = rewards

        batch_size = len(self) if self.batch_size <= 0 else self.batch_size
        indices = np.random.choice(len(self), batch_size, replace=False)

        state = torch.cat(self._states, dim=0)[indices]
        action = torch.cat(self._actions, dim=0)[indices]
        prob = torch.cat(self._probs, dim=0)[indices]

        reward = torch.tensor(self._rewards, dtype=torch.float32).unsqueeze(1)[indices]
        done = torch.tensor(self._dones, dtype=torch.float32).unsqueeze(1)[indices]

        experience = Experience(
            state=state,
            prob=prob,
            action=action, reward=reward, done=done).to(self.device)
        return experience

    def __len__(self):
        return len(self._states)

    def clear(self):
        del self._states[:]
        del self._actions[:]
        del self._rewards[:]
        del self._dones[:]
        del self._probs[:]
        self.rollout_reward = False


    # @abstractmethod
    # def _batch_stack(self, batch):
    #     return batch


