import rldev

import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, MultivariateNormal

from rllib.buffer import ReplayBuffer
from rllib.utils import init_weights, soft_update
from rllib.template import MethodSingleAgent, Model
from rllib.template.model import FeatureExtractor, FeatureMapper


class SAC(MethodSingleAgent):
    gamma = 0.99

    reward_scale = 1.0
    target_entropy = None
    alpha_init = 1.0

    lr_critic = 0.0003
    lr_actor = 0.0003
    lr_tune = 0.0003

    tau = 0.005

    replay_buffer_size = 1000000
    buffer_size = 30000
    num_iters = 100
    batch_size = 256

    save_model_interval = 200

    def __init__(self, config, writer, tag_name='method'):
        super().__init__(config, writer, tag_name)

        ### params
        self.batch_size = config.get('batch_size', self.batch_size)
        self.replay_buffer_size = config.get('replay_buffer_size', self.replay_buffer_size)
        self.buffer_size = config.get('buffer_size', self.buffer_size)
        self.buffer_size = config.get('num_iters', self.num_iters)
        assert self.buffer_size <= self.replay_buffer_size

        self.reward_scale = config.get('reward_scale', self.reward_scale)

        self.lr_critic = config.get('lr_critic', self.lr_critic)
        self.lr_actor = config.get('lr_actor', self.lr_actor)
        self.lr_tune = config.get('lr_tune', self.lr_tune)

        ### model
        self.critic: Critic = config.get('net_critic', Critic)(config).to(self.device)
        self.actor: Actor = config.get('net_actor', Actor)(config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.models_to_save = [self.critic, self.actor]

        self.critic_optimizer= Adam(self.critic.parameters(), lr=self.lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizers = [self.critic_optimizer, self.actor_optimizer]
        self.critic_loss = nn.MSELoss()

        ### automatic entropy tuning
        if self.target_entropy == None:
            self.target_entropy = -np.prod((self.dim_action,)).item()
        self.log_alpha = torch.full((), np.log(self.alpha_init), requires_grad=True, dtype=self.dtype, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_tune)

        self.buffer: ReplayBuffer = config.get('buffer', ReplayBuffer)(config, self.replay_buffer_size, self.batch_size, self.device)


    def update_parameters(self):
        if len(self.buffer) < self.buffer_size:  ### start timestamps
            return
        
        for _ in range(self.num_iters):
            self.update_parameters_one_step()
        return
        
    def update_parameters_one_step(self):
        self.update_parameters_start()
        self.writer.add_scalar(f'{self.tag_name}/replay_buffer_size', len(self.buffer), self.step_update)

        '''load data batch'''
        print('---------------- here')
        experience = self.buffer.sample()
        state = experience.state
        action = experience.action_data.action
        next_state = experience.next_state
        reward = experience.reward *self.reward_scale
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
        alpha_loss = self.log_alpha.exp() * (-logprob.mean() - self.target_entropy).detach()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        self.writer.add_scalar(f'{self.tag_name}/loss_critic', critic_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/loss_actor', actor_loss.detach().item(), self.step_update)
        self.writer.add_scalar(f'{self.tag_name}/alpha', self.alpha.detach().item(), self.step_update)

        self._update_model()
        if self.step_update % self.save_model_interval == 0:
            self._save_model()
        
        self.update_callback(locals())
        return


    @torch.no_grad()
    def select_action(self, state):
        self.select_action_start()

        if self.step_select < self.buffer_size:
            action = torch.Tensor(1,self.dim_action).uniform_(-1,1)
        else:
            action, _, _ = self.actor.sample(state.to(self.device))
            action = action.cpu()
        return rldev.Data(action=action)


    def _update_model(self):
        # print('[update_parameters] soft update')
        soft_update(self.critic_target, self.critic, self.tau)



class Actor(Model):
    logstd_min = -5
    logstd_max = 1

    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)

        self.fe = config.get('net_actor_fe', FeatureExtractor)(config, 0)
        self.mean = config.get('net_actor_fm', FeatureMapper)(config, 0, self.fe.dim_feature, config.dim_action)
        self.mean_no = nn.Tanh()
        self.std = copy.deepcopy(self.mean)
        self.std_no = nn.Tanh()
        self.apply(init_weights)
    
    def forward(self, state):
        x = self.fe(state)
        mean = self.mean_no(self.mean(x))
        logstd = self.std_no(self.std(x))
        logstd = (self.logstd_max-self.logstd_min) * logstd + (self.logstd_max+self.logstd_min)
        return mean, logstd *0.5


    def sample(self, state):
        mean, logstd = self(state)

        cov = torch.diag_embed( torch.exp(logstd) )
        dist = MultivariateNormal(mean, cov)
        u = dist.rsample()


        # if mean.shape[0] == 1:
        #     print('    policy entropy: ', dist.entropy().detach().cpu())
        #     print('    policy mean:    ', mean.detach().cpu())
        #     print('    policy std:     ', torch.exp(logstd).detach().cpu())
        #     print()



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
    def __init__(self, config, model_id=0):
        super().__init__(config, model_id)

        self.fe = config.get('net_critic_fe', FeatureExtractor)(config, 0)
        self.fm1 = config.get('net_critic_fm', FeatureMapper)(config, 0, self.fe.dim_feature+config.dim_action, 1)
        self.fm2 = copy.deepcopy(self.fm1)
        self.apply(init_weights)

    def forward(self, state, action):
        x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x), self.fm2(x)
    
    def q1(self, state, action):
        x = self.fe(state)
        x = torch.cat([x, action], 1)
        return self.fm1(x)

