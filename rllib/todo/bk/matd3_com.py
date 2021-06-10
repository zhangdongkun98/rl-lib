import carla_utils as cu

import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=3, threshold=np.inf, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=3, linewidth=65536, suppress=True)

from utils import soft_update, init_weights, to_device, DeepSet, Model
from utils.method_multi_agent import Method
from utils.replay_buffer import ReplayBufferMultiAgent



class MATD3(Method):
    def __init__(self, config, device, path_pack):
        super(MATD3, self).__init__(config, device, path_pack)

        '''param'''
        self.policy_noise, self.noise_clip = config.policy_noise, config.noise_clip
        self.policy_freq = config.policy_freq

        '''network'''
        self.critics = [Critic(config, vi, self.dim_state, self.dim_action).to(device) for vi in range(self.num_vehicles)]
        self.actors = [Actor(config, vi, self.dim_state, self.dim_action).to(device) for vi in range(self.num_vehicles)]
        self.critics_target = deepcopy(self.critics)
        self.actors_target = deepcopy(self.actors)
        self.models_to_load = self.critics + self.actors + self.critics_target + self.actors_target
        self.models_to_save = self.critics + self.actors
        if config.load_model: self._load_model()
        self.executor = ThreadPoolExecutor(self.num_vehicles)

        self.critics_optimizer= [Adam(model.parameters(), lr=5e-4) for model in self.critics]
        self.actors_optimizer = [Adam(model.parameters(), lr=1e-4) for model in self.actors]
        self.critic_loss = nn.MSELoss()
        self._replay_buffer = ReplayBufferMultiAgent(config, device)


    def update_policy(self):
        self.train_step += 1
        if self.train_step % 100 == 0:
            print('[update_policy] buffer size: ', len(self._replay_buffer))


        self.c_loss = {vi: 0.0 for vi in range(self.num_vehicles)}
        self.a_loss = {vi: 0.0 for vi in range(self.num_vehicles)}

        '''load data'''
        self.experiences = self._replay_buffer.sample()

        # for vi in range(self.num_vehicles): self._train_agent(vi); #print('[warning !!! single thread.]')
        vis = list(range(self.num_vehicles))
        [_ for _ in self.executor.map(self._train_agent, vis)]
        if self.train_step % self.policy_freq == 0: self._update_model()

        del self.experiences

        if self.train_step % 300 == 0: self._save_model()
        if self.train_step % 300 == 0: torch.cuda.empty_cache()
        return self.c_loss, self.a_loss

    def _train_agent(self, vi):
        critic, critic_target = self.critics[vi], self.critics_target[vi]
        actor = self.actors[vi]
        critic_optimizer, actor_optimizer = self.critics_optimizer[vi], self.actors_optimizer[vi]

        import time

        t1 = time.time()

        experiences = self.experiences

        state = experiences[vi].states
        mask_index = torch.where(state.mask[:,vi] == -1)

        actions = experiences[vi].actions
        reward  = experiences[vi].rewards[mask_index][:,vi].unsqueeze(1)
        done    = experiences[vi].dones[mask_index][:,vi].unsqueeze(1)
        states = [e.states for e in experiences]
        next_states = [e.next_states for e in experiences]

        

        

        t2 = time.time()
        



        '''critic'''
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = [self.actors_target[ns.vi](ns) for ns in next_states]
            next_actions = torch.cat(next_actions, dim=1)
            next_actions = (next_actions + noise).clamp(-1,1)
            next_actions[torch.where(actions == np.inf)] = np.inf

            target_q1, target_q2 = critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (target_q * self.gamma * (1-done)).detach()

        current_q1, current_q2 = critic(states, actions)
        critic_loss = self.critic_loss(current_q1, target_q) + self.critic_loss(current_q2, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        self.c_loss[vi] = critic_loss.detach().item()


        t3 = time.time()

        '''actor'''
        if self.train_step % self.policy_freq == 0:
            action = actor(state)
            actions_new = actions.clone()
            actions_new[:,vi] = action.squeeze()
            actor_loss = -critic.q1(states, actions_new.detach(), action).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            self.a_loss[vi] = actor_loss.detach().item()
        
        t4 = time.time()


        print('train one: ', t2-t1, t3-t2, t4-t3)
        return


    @torch.no_grad()
    def _select_actions_train(self, states):
        self.total_timesteps += 1
        if self.total_timesteps < self.start_timesteps:
            actions_normal = torch.from_numpy(np.random.uniform(-1,1, len(states)).astype(np.float32))
        else:
            std = torch.ones(len(states)) * 0.1
            noise = torch.normal(0, std)
            actions_normal = [self.actors[state.vi](to_device(state, self.device)).item() for state in states]
            actions_normal = (torch.tensor(actions_normal) + noise).clamp(-1,1)
        return actions_normal
        
    @torch.no_grad()
    def _select_actions_eval(self, states):
        actions_normal = [self.actors[state.vi](to_device(state, self.device)).item() for state in states]
        actions_normal = torch.tensor(actions_normal)
        return actions_normal
    
    def _update_model(self):
        for vi in range(self.num_vehicles):
            soft_update(self.critics_target[vi], self.critics[vi], self.tau)
            soft_update(self.actors_target[vi], self.actors[vi], self.tau)



class Critic(Model):
    def __init__(self, config, model_id, dim_state, dim_action):
        super(Critic, self).__init__(config, model_id)
        
        self.dim_ds = 64
        self.deepset_s = DeepSet(dim_state, self.dim_ds)

        dim_dynamic_feature = 128
        self.deepset_v = DeepSet(dim_state + self.dim_ds + dim_action, dim_dynamic_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.fc2 = deepcopy(self.fc1)
        self.apply(init_weights)

    def forward(self, states, actions):
        state = states[self.model_id]

        features = []
        for so in states:
            so_dynamic, so_mask = so.dynamic, so.mask
            so_mask_index = torch.where((so_mask[:,self.model_id] == 1) & (so_mask[:,so.vi] != -3))
            so_dynamic_valid, so_mask_valid = so_dynamic[so_mask_index], so_mask[so_mask_index]
            fo = torch.ones((so_mask.shape[0], self.dim_ds), dtype=so_dynamic.dtype, device=so_dynamic.device) * np.inf
            fo[so_mask_index] = self.deepset_s(so_dynamic_valid, so_mask_valid)
            features.append( torch.cat([so.fixed, fo, actions[:,so.vi].unsqueeze(1)], dim=1) )
        states_dynamic = torch.stack(features, dim=1)

        mask_index = torch.where(state.mask[:,self.model_id] == -1)
        x_dynamic = self.deepset_v(states_dynamic[mask_index], state.mask[mask_index])
        x = torch.cat([state.fixed[mask_index], x_dynamic, actions[:,self.model_id].unsqueeze(1)[mask_index]], dim=1)
        return self.fc1(x), self.fc2(x)

    def q1(self, states, actions, action):
        state = states[self.model_id]

        features = []
        for so in states:
            so_dynamic, so_mask = so.dynamic, so.mask
            so_mask_index = torch.where((so_mask[:,self.model_id] == 1) & (so_mask[:,so.vi] != -3))
            so_dynamic_valid, so_mask_valid = so_dynamic[so_mask_index], so_mask[so_mask_index]
            fo = torch.ones((so_mask.shape[0], self.dim_ds), dtype=so_dynamic.dtype, device=so_dynamic.device) * np.inf
            fo[so_mask_index] = self.deepset_s(so_dynamic_valid, so_mask_valid)
            features.append( torch.cat([so.fixed, fo, actions[:,so.vi].unsqueeze(1)], dim=1) )
        states_dynamic = torch.stack(features, dim=1)

        mask_index = torch.where(state.mask[:,self.model_id] == -1)
        x_dynamic = self.deepset_v(states_dynamic[mask_index], state.mask[mask_index])
        x = torch.cat([state.fixed[mask_index], x_dynamic, action[mask_index]], dim=1)
        return self.fc1(x)


class Actor(Model):
    def __init__(self, config, model_id, dim_state, dim_action):
        super(Actor, self).__init__(config, model_id)
        self.dim_action = dim_action
        
        dim_dynamic_feature = 64
        self.deepset = DeepSet(dim_state, dim_dynamic_feature)
        self.fc = nn.Sequential(
            nn.Linear(dim_state + dim_dynamic_feature, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, dim_action), nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, state):
        mask_index = torch.where(state.mask[:,self.model_id] == -1)
        x_fixed, x_dynamic, x_mask = state.fixed[mask_index], state.dynamic[mask_index], state.mask[mask_index]
        
        x_dynamic = self.deepset(x_dynamic, x_mask)

        x = torch.cat([x_fixed, x_dynamic], dim=1)
        result = torch.ones((state.mask.shape[0], self.dim_action), device=state.mask.device) * np.inf
        result[mask_index] = self.fc(x)
        return result

