import carla_utils as cu

import numpy as np
from copy import deepcopy
import random
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
torch.set_printoptions(precision=3, threshold=np.inf, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)
np.set_printoptions(precision=3, linewidth=65536, suppress=True, threshold=sys.maxsize)

from utils import soft_update, init_weights, DeepSet
from utils.method_multi_agent import Method, Model
from utils.replay_buffer import ReplayBuffer



class MADDPG(Method):
    def __init__(self, device, config, path_pack):
        super(MADDPG, self).__init__(device, config, path_pack)

        '''param'''
        self.gamma = config.gamma
        self.tau = config.tau

        '''network'''
        self.critics = [Critic(vehicle_index, config).to(device) for vehicle_index in range(self.num_vehicles)]
        self.actors = [Actor(vehicle_index, config).to(device) for vehicle_index in range(self.num_vehicles)]
        self.critics_target = deepcopy(self.critics)
        self.actors_target = deepcopy(self.actors)
        self.models_to_load = self.critics + self.actors + self.critics_target + self.actors_target
        self.models_to_save = self.critics + self.actors
        if config.load_model is True: self._load_model()

        self.critics_optimizer= [Adam(model.parameters(), lr=5e-4) for model in self.critics]
        self.actors_optimizer = [Adam(model.parameters(), lr=1e-4) for model in self.actors]

        self.critic_loss = nn.MSELoss()

        self.memory = ReplayBuffer(config, device)

        self.epsilon_prob = config.epsilon_prob_start
        self.decay_rate = (config.epsilon_prob_end / config.epsilon_prob_start) ** (1/config.n_episode)
        self.decay_rate = config.decay_rate


    def update_policy(self):
        self.train_step += 1
        if self.train_step % 100 == 0:
            print('[update_policy] buffer size: ', len(self.memory))

        c_loss = {vehicle_index: 0.0 for vehicle_index in range(self.num_vehicles)}
        a_loss = {vehicle_index: 0.0 for vehicle_index in range(self.num_vehicles)}
        for vehicle_index in range(self.num_vehicles):
            self._train_agent(vehicle_index, c_loss, a_loss)

        self._update_model()
        if self.train_step % 1000 == 0: self._save_model()
        if self.train_step % 1000 == 0: torch.cuda.empty_cache()
        return c_loss, a_loss

    def _train_agent(self, vehicle_index, c_loss, a_loss):
        '''network to be updated'''
        critic, critic_target = self.critics[vehicle_index], self.critics_target[vehicle_index]
        actor = self.actors[vehicle_index]
        critic_optimizer, actor_optimizer = self.critics_optimizer[vehicle_index], self.actors_optimizer[vehicle_index]

        '''load data batch'''
        experience_batch = self.memory.sample()
        states_batch = experience_batch.states
        actions_batch = experience_batch.actions
        next_states_batch = experience_batch.next_states
        rewards_batch = experience_batch.rewards
        dones_batch = experience_batch.dones
        masks_batch = experience_batch.masks

        mask_index = torch.where(masks_batch[:,vehicle_index] == 1)
        if sum(masks_batch[:,vehicle_index]) == 0: return

        '''critic'''
        with torch.no_grad():
            ### get next actions
            next_actions_batch = torch.ones(actions_batch.shape, device=self.device) * np.inf
            for vehicle_num in range(self.num_vehicles):
                if sum(masks_batch[:,vehicle_num]) == 0: continue

                other_mask_index = torch.where(masks_batch[:,vehicle_num] == 1)
                index = next_states_batch.indices[:,vehicle_num][other_mask_index]

                tmp_index = index.clone()
                tmp_index[torch.where(tmp_index == -1)] = 0
                obsv = torch.gather(next_states_batch.items[other_mask_index], 1, tmp_index.unsqueeze(-1).repeat(1,1,self.dim_state))
                ### Size([valid_batch_size, self.num_vehicles, self.dim_state])

                tmp_index = index.clone()
                tmp_index[torch.where(tmp_index > -1)] = 1
                tmp_index[torch.where(tmp_index == -1)] = 0
                lengths = tmp_index.sum(dim=1)
                ### Size([valid_batch_size])

                next_actions_batch[:,vehicle_num,:][other_mask_index] = self.actors_target[vehicle_num](obsv, lengths)
            
            target_q = critic_target(next_states_batch, next_actions_batch, masks_batch)
            target_q = (target_q * self.gamma * (1-dones_batch[:,vehicle_index][mask_index]).unsqueeze(1)).detach() + rewards_batch[:,vehicle_index][mask_index].unsqueeze(1)

        current_q = critic(states_batch, actions_batch, masks_batch)
        critic_loss = self.critic_loss(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        '''actor'''
        actions = actions_batch.clone()

        index = states_batch.indices[:,vehicle_index][mask_index]
        tmp_index = index.clone()
        tmp_index[torch.where(tmp_index == -1)] = 0
        obsv = torch.gather(states_batch.items[mask_index], 1, tmp_index.unsqueeze(-1).repeat(1,1,self.dim_state))

        tmp_index = index.clone()
        tmp_index[torch.where(tmp_index > -1)] = 1
        tmp_index[torch.where(tmp_index == -1)] = 0
        lengths = tmp_index.sum(dim=1)
        action_prob = actor(obsv, lengths)
        actions[:,vehicle_index,:][mask_index] = action_prob
        actor_loss = -critic(states_batch, actions, masks_batch).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        '''loss'''
        c_loss[vehicle_index] += critic_loss.detach().item()
        a_loss[vehicle_index] += actor_loss.detach().item()

        # output = np.hstack((
        #     cu.basic.onehot2int(actions_batch[:,vehicle_index][mask_index]).unsqueeze(1).data.cpu().numpy(),
        #     states_batch.items[mask_index][:,vehicle_index,5:7].data.cpu().numpy(),
        #     dones_batch[:,vehicle_index][mask_index].unsqueeze(1).data.cpu().numpy(),
        #     rewards_batch[:,vehicle_index][mask_index].unsqueeze(1).data.cpu().numpy(),
        #     current_q.data.cpu().numpy(),
        #     target_q.data.cpu().numpy(),
        #     action_prob.data.cpu().numpy(),
        # ))
        # np.savetxt(join(self.path_pack.output_path, str(self.train_step)+'_'+str(vehicle_index)+'.txt'), output, delimiter='   ', fmt='%7f')

        # print('time: ', tick2-tick1, tick3-tick2, tick4-tick3)


        return


    @torch.no_grad()
    def _select_actions_train(self, states, masks):
        num_vehicles = masks.sum()
        items = states.items.to(self.device)  ### torch.Size([num_vehicles, dim_state])
        actions = torch.zeros(num_vehicles, dtype=torch.int64)
        actions_prob = torch.zeros((num_vehicles, self.dim_action), dtype=torch.float32)
        masks_index = torch.where(masks == 1)
        for i, index in enumerate(states.indices[masks_index]):
            if random.random() < self.epsilon_prob:
                action = random.choice(list(range(self.dim_action)))
                action_prob = cu.basic.int2onehot(torch.tensor(action, dtype=torch.float32), self.dim_action)
            else:
                obsv = items[index[torch.where(index > -1)]].unsqueeze(0)
                action_prob = self.actors[index[0]](obsv).squeeze()
                action = torch.argmax(action_prob.detach()).cpu().item()
            actions[i] = int(action)
            actions_prob[i] = action_prob
        self.epsilon_prob *= self.decay_rate
        return actions, actions_prob
    
    
    @torch.no_grad()
    def _select_actions_eval(self, states, masks):
        return
    

    def _update_model(self):
        # print('[update_policy] soft update')
        for vehicle_index in range(self.num_vehicles):
            soft_update(self.critics_target[vehicle_index], self.critics[vehicle_index], self.tau)
            soft_update(self.actors_target[vehicle_index], self.actors[vehicle_index], self.tau)




class Critic(Model):
    def __init__(self, model_id, config):
        super(Critic, self).__init__()
        self.file_dir = os.path.dirname(__file__)
        self.model_id = model_id
        self.num_vehicles = config.num_vehicles
        self.dim_state = config.dim_state
        self.dim_action = config.dim_action

        self.fc = nn.Sequential(
            nn.Linear(self.rnn_layer*self.dim_rnn_hidden, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.apply(init_weights)
  

    def forward(self, states, actions, masks):
        return


class Actor(Model):
    def __init__(self, model_id, config):
        super(Actor, self).__init__()
        self.file_dir = os.path.dirname(__file__)
        self.model_id = model_id
        self.dim_state = config.dim_state
        self.dim_action = config.dim_action
        self.dim_rnn_hidden = 128
        self.rnn_layer = 1


        self.fc = nn.Sequential(
            nn.Linear(self.rnn_layer*self.dim_rnn_hidden, 256), nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64), nn.LeakyReLU(inplace=True),
            nn.Linear(64, self.dim_action), nn.Softmax(dim=1),
        )

        self.apply(init_weights)

    def forward(self, observation, lengths=None):
        return 
    