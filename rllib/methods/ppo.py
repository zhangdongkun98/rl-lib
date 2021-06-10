
import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .tools import init_weights

import utils

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]



class ActorCritic(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(dim_state, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, dim_action), nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(dim_state, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.apply(init_weights)
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO(object):
    def __init__(self, config, device):
        dim_state, dim_action = config.dim_state, config.dim_action
        lr, betas = config.lr, eval(config.betas)
        self.gamma = config.gamma

        ### ppo
        self.epsilon_clip = config.epsilon_clip
        self.K_epochs = config.K_epochs
        self.wv, self.we = config.weight_value, config.weight_entropy
        
        self.policy = ActorCritic(dim_state, dim_action).to(device)
        self.policy_old = copy.deepcopy(self.policy)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.MseLoss = nn.MSELoss()
        self.memory = Memory()

        self.train_step = -1

    
    def update(self, writer):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        

        ### -------------------------------
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            self.train_step += 1

            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.wv*self.MseLoss(state_values, rewards) - self.we*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            writer.add_scalar('loss/loss', loss.detach().item(), self.train_step)
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()

