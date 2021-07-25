
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal

from .buffer import ReplayBufferOffPolicy as ReplayBuffer
from .utils import init_weights, soft_update
from .template import MethodSingleAgent, Model


class SAC(MethodSingleAgent):
    gamma = 0.99

    lr_critic = 0.0003
    lr_actor = 0.0003
    lr_tune = 0.0003

    tau = 0.005

    buffer_size = 1000000
    batch_size = 256

    start_timesteps = 30000

    save_model_interval = 1000

    def __init__(self, config, writer):
        super(SAC, self).__init__(config, writer)

        self.critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor: Actor = config.get('net_actor', Actor)(config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_loss = nn.MSELoss()

        ### automatic entropy tuning
        self.alpha = 0.2
        self.target_entropy = -torch.prod(torch.Tensor(self.dim_action).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_tune)

        self._memory: ReplayBuffer = config.get('buffer', ReplayBuffer)(self.buffer_size, self.batch_size, self.device)


    def update_parameters(self):
        if len(self._memory) < self.start_timesteps:
            return
        super().update_parameters()

        '''load data batch'''
        experience = self._memory.sample()
        state = experience.state
        action = experience.action
        next_state = experience.next_state
        reward = experience.reward
        done = experience.done

        '''critic'''
        with torch.no_grad():
            next_action, next_logprob, _ = self.actor.sample(next_state)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logprob
            target_q = reward + self.gamma * (1-done) * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''actor'''
        action, logprob, _ = self.actor.sample(state)
        actor_loss = (-self.critic.q1(state, action) + self.alpha * logprob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        '''automatic entropy tuning'''
        alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        self.writer.add_scalar('loss/c_loss', critic_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/a_loss', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar('loss/alpha', self.alpha.detach().item(), self.step_update)

        self._update_model()
        if self.step_update % self.save_model_interval == 0: self._save_model()
        return


    @torch.no_grad()
    def select_action(self, state):
        super().select_action()

        if self.step_select < self.start_timesteps:
            action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            _, _, action = self.actor.sample(state.to(self.device))
        return action


    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.critic_target, self.critic, self.tau)
        # soft_update(self.actor_target, self.actor, self.tau)



class Actor(Model):
    logstd_min = -5
    logstd_max = 5

    def __init__(self, config):
        super(Actor, self).__init__(config, model_id=0)

        self.mean = nn.Sequential(
            nn.Linear(config.dim_state, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, config.dim_action), nn.Tanh(),
        )
        self.std = copy.deepcopy(self.mean)
        self.apply(init_weights)
    
    def forward(self, state):
        mean = self.mean(state)
        logstd = self.std(state)
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5


    def sample(self, state):
        mean, logstd = self(state)

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()

        ### Enforcing Action Bound
        action = torch.tanh(u)
        logprob = dist.log_prob(u).unsqueeze(1) \
                - torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)

        return action, logprob, mean


    def sample_deprecated(self, state):
        mean, logstd = self(state)

        dist = Normal(mean, torch.exp(logstd))
        u = dist.rsample()

        ### Enforcing Action Bound
        action = torch.tanh(u)
        logprob = dist.log_prob(u) - torch.log(1-action.pow(2) + 1e-6)
        logprob = logprob.sum(dim=1, keepdim=True)

        return action, logprob, mean



class Critic(Model):
    def __init__(self, config):
        super(Critic, self).__init__(config, model_id=0)

        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_state+config.dim_action, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.fc2 = copy.deepcopy(self.fc1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x), self.fc2(x)
    
    def q1(self, state, action):
        x = torch.cat([state, action], 1)
        return self.fc1(x)

