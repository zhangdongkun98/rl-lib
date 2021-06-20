

import copy

import torch
import torch.nn as nn
from torch.optim import Adam

from .utils import init_weights, soft_update
from .template import MethodSingleAgent, Model, ReplayBufferSingleAgent, Experience


class DDPG(MethodSingleAgent):
    gamma = 0.99
    
    lr_critic = 0.0003
    lr_actor = 0.0003

    tau = 0.005

    buffer_size = 1000000
    batch_size = 256

    policy_freq = 2
    explore_noise = 0.1
    policy_noise = 0.2
    noise_clip = 0.5

    start_timesteps = 30000

    def __init__(self, config, writer):
        super(DDPG, self).__init__(config, writer)

        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor = config.get('net_actor', Actor)(config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()

        self._replay_buffer = config.get('buffer', ReplayBuffer)(self.buffer_size, self.batch_size, config.device)


    def update_policy(self):
        if len(self._replay_buffer) < self.start_timesteps:
            return
        super().update_policy()

        '''load data batch'''
        experience = self._replay_buffer.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        '''critic'''
        with torch.no_grad():
            target_q = self.critic_target(next_state, self.actor_target(next_state))
            target_q = reward + self.gamma * (1-done) * target_q

        current_q = self.critic(state, action)
        critic_loss = self.critic_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self._update_model()

        self.writer.add_scalar('loss/a_loss', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/c_loss', critic_loss.detach().item(), self.step_update)
        
        # if self.step_update % 200 == 0: self._save_model()
        return


    @torch.no_grad()
    def select_action(self, state):
        super().select_action()

        if self.step_select < self.start_timesteps:
            action_normal = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            noise = torch.normal(0, self.explore_noise, size=(1,self.dim_action))
            action_normal = self.actor(state.to(self.device))
            action_normal = (action_normal.cpu() + noise).clamp(-1,1)
        return action_normal

    def _update_model(self):
        # print('[update_policy] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)



class ReplayBuffer(ReplayBufferSingleAgent):
    def _batch_stack(self, batch):
        state, action, next_state, reward, done = [], [], [], [], []
        for e in batch:
            state.append(e.state)
            action.append(e.action)
            next_state.append(e.next_state)
            reward.append(e.reward)
            done.append(e.done)

        state = torch.cat(state, dim=0)
        action = torch.cat(action, dim=0)
        next_state = torch.cat(next_state, dim=0)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        experience = Experience(
            state=state,
            next_state=next_state,
            action=action, reward=reward, done=done).to(self.device)
        return experience



class Actor(Model):
    def __init__(self, config):
        super(Actor, self).__init__(config, model_id=0)

        self.fc = nn.Sequential(
            nn.Linear(config.dim_state, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, config.dim_action), nn.Tanh(),
        )
        self.apply(init_weights)
    
    def forward(self, state):
        return self.fc(state)


class Critic(Model):
    def __init__(self, config):
        super(Critic, self).__init__(config, model_id=0)

        self.fc = nn.Sequential(
            nn.Linear(config.dim_state+config.dim_action, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc(x)
    


