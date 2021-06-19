
import copy
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from .utils import init_weights
from .template import MethodSingleAgent, Model, ReplayBufferSingleAgent, Experience

class PPO(MethodSingleAgent):
    gamma = 0.99
    K_epochs = 4
    epsilon_clip = 0.2
    weight_value = 1.0
    weight_entropy = 0.01
    # weight_entropy = 0.00

    lr = 0.002
    betas = (0.9, 0.999)

    buffer_size = 2000

    std_action = 0.5

    def __init__(self, config, writer):
        super(PPO, self).__init__(config, writer)

        self.policy: Union[ActorCriticDiscrete, ActorCriticContinuous] = config.net_ac(config, self.std_action).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.models_to_save = [self.policy]

        self.optimizer = Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        # self.critic_loss = nn.MSELoss(reduction='none')
        self.critic_loss = nn.MSELoss()
        self._replay_buffer = Memory()


    def update_policy(self):
        if len(self._replay_buffer) < self.buffer_size:
            return
        super().update_policy()

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self._replay_buffer.rewards), reversed(self._replay_buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.cat(self._replay_buffer.states).detach().to(self.device)
        old_actions = torch.cat(self._replay_buffer.actions).detach().to(self.device)
        old_logprobs = torch.cat(self._replay_buffer.logprobs).detach().to(self.device)

        # import pdb; pdb.set_trace()

        for _ in range(self.K_epochs):
            self.step_train += 1

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # import pdb; pdb.set_trace()

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages

            loss =  - torch.min(surr1, surr2) \
                    - self.weight_entropy*dist_entropy \
                    + self.weight_value*self.critic_loss(state_values, rewards)


            # import pdb; pdb.set_trace()

            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('loss/loss', loss.detach().item(), self.step_train)
        
        if self.step_update % 20 == 0: self._save_model()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self._replay_buffer.clear_memory()

    @torch.no_grad()   ## ! maybe important
    def select_action(self, state):
        super().select_action()
        action, log_prob = self.policy_old.act(state.to(self.device))
        self._replay_buffer.logprobs.append(log_prob)
        return action



class ActorCriticDiscrete(Model):
    def __init__(self, config, std_action):
        super(ActorCriticDiscrete, self).__init__(config, model_id=0)

        self.actor = self.Actor(config)
        self.critic = self.Critic(config)
        self.apply(init_weights)


    def forward(self):
        raise NotImplementedError

    def act(self, state):
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
    def __init__(self, config, std_action):
        super(ActorCriticContinuous, self).__init__(config, model_id=0)

        self.actor = self.Actor(config)
        self.critic = self.Critic(config)
        self.apply(init_weights)

        self.var = torch.full((config.dim_action,), std_action**2).to(self.device)
        self.cov = torch.diag(self.var)
        
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, self.cov)
        action = dist.sample()
        return action, dist.log_prob(action).unsqueeze(1)
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        var = self.var.expand_as(action_mean)
        cov = torch.diag_embed(var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov)
        action_logprobs = dist.log_prob(action).unsqueeze(1)
        dist_entropy = dist.entropy().unsqueeze(1)

        state_value = self.critic(state)
        return action_logprobs, state_value, dist_entropy


    class Actor(Model):
        def __init__(self, config):
            super().__init__(config, model_id=0)

            self.fc = nn.Sequential(
                nn.Linear(config.dim_state, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, config.dim_action), nn.Tanh(),
            )
        
        def forward(self, state):
            return self.fc(state)
    
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



class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def push(self, experience):
        self.actions.append(experience.action)
        self.states.append(experience.state)
        self.rewards.append(experience.reward)
        self.dones.append(experience.done)
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
    
    def __len__(self):
        return len(self.actions)




