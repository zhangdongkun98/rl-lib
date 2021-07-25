
import copy
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from .buffer import RolloutBufferOnPolicy as RolloutBuffer
from .utils import init_weights, hard_update
from .template import MethodSingleAgent, Model


class PPO(MethodSingleAgent):
    gamma = 0.99
    epsilon_clip = 0.2
    weight_value = 1.0
    weight_entropy = 0.001
    weight_entropy = 0.01

    lr = 0.002
    lr = 0.0003
    betas = (0.9, 0.999)

    K_epochs = 4
    K_epochs = 10
    buffer_size = 2000
    batch_size = 0
    batch_size = 32

    save_model_interval = 20

    def __init__(self, config, writer):
        super(PPO, self).__init__(config, writer)

        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        self.critic_loss = nn.MSELoss()
        self._memory: RolloutBuffer = config.get('buffer', RolloutBuffer)(self.device, self.batch_size)


    def update_parameters(self):
        if len(self._memory) < self.buffer_size:
            return
        super().update_parameters()

        for _ in range(self.K_epochs):
            self.step_train += 1

            experience = self._memory.sample(self.gamma)

            old_states = experience.state
            old_actions = experience.action
            old_logprobs = experience.prob

            rewards = experience.reward
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages

            loss =  - torch.min(surr1, surr2) \
                    - self.weight_entropy*dist_entropy \
                    + self.weight_value*self.critic_loss(state_values, rewards)

            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.optimizer.step()

            self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_train)
        
        if self.step_update % self.save_model_interval == 0: self._save_model()

        ### Copy new weights into old policy:
        hard_update(self.policy_old, self.policy)
        self._memory.clear()

    @torch.no_grad()
    def select_action(self, state):
        super().select_action()
        action, logprob = self.policy_old(state.to(self.device))
        self._memory.push_prob(logprob)
        return action



class ActorCriticDiscrete(Model):
    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.actor = self.Actor(config)
        self.critic = self.Critic(config)
        self.apply(init_weights)


    def forward(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.unsqueeze(1), dist.log_prob(action).unsqueeze(1)

    def evaluate(self, state, action):
        action_probs = self.actor(state)

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action.squeeze()).unsqueeze(1)
        dist_entropy = dist.entropy().unsqueeze(1)

        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy

    class Actor(Model):
        def __init__(self, config):
            super().__init__(config, model_id=0)

            self.fc = nn.Sequential(
                nn.Linear(config.dim_state, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, config.dim_action), nn.Softmax(dim=1),
            )
        
        def forward(self, state):
            return self.fc(state)
    
    class Critic(Model):
        def __init__(self, config):
            super().__init__(config, model_id=0)

            self.fc = nn.Sequential(
                nn.Linear(config.dim_state, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 1),
            )
        
        def forward(self, state):
            return self.fc(state)


class ActorCriticContinuous(Model):
    def __init__(self, config):
        super().__init__(config, model_id=0)

        self.actor = self.Actor(config)
        self.critic = self.Critic(config)
        self.apply(init_weights)
        
    def forward(self, state):
        action_mean, action_logstd = self.actor(state)

        cov = torch.diag_embed( torch.exp(action_logstd) )
        dist = MultivariateNormal(action_mean, cov)
        action = dist.sample()
        return action, dist.log_prob(action).unsqueeze(1)
    
    def evaluate(self, state, action):
        action_mean, action_logstd = self.actor(state)

        cov = torch.diag_embed( torch.exp(action_logstd) )
        dist = MultivariateNormal(action_mean, cov)
        logprob = dist.log_prob(action).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1)

        value = self.critic(state)
        return logprob, value, entropy


    class Actor(Model):
        logstd_min = -1
        logstd_max = 1

        def __init__(self, config):
            super().__init__(config, model_id=0)

            self.mean = nn.Sequential(
                nn.Linear(config.dim_state, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, config.dim_action), nn.Tanh(),
            )
            self.std = copy.deepcopy(self.mean)
        
        def forward(self, state):
            mean = self.mean(state)
            logstd = self.std(state)
            logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
            return mean, logstd *0.5
    
    class Critic(Model):
        def __init__(self, config):
            super().__init__(config, model_id=0)

            self.fc = nn.Sequential(
                nn.Linear(config.dim_state, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1),
            )
        
        def forward(self, state):
            return self.fc(state)

