import torch
import torch.nn as nn
import os
import numpy as np
from config import Config
from sub_q_net import SubNet
from mix_net import MixNet


class QMixPolicy(object):
    def __init__(self):
        self.conf = Config()
        self.device = self.conf.device
        self.n_actions = self.conf.N_ACTIONS
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_state = self.conf.obs_shape
        self.memory = np.zeros((self.conf.MEMORY_CAPACITY, self.conf.N_STATES * 16 + 8 + 2))

        self.eval_sub_net = SubNet().to(self.device)
        self.target_sub_net = SubNet().to(self.device)
        self.eval_mix_net = MixNet().to(self.device)
        self.target_mix_net = MixNet().to(self.device)

        self.eval_parameters = list(self.eval_sub_net.parameters()) + list(self.eval_mix_net.parameters())
        self.loss = 0.
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.conf.LR)

        self.epsilon = self.conf.epsilon
        self.memory_counter = 0
        self.train_step = 0

    def choose_actions(self, states, eval=False):
        actions = []
        if eval:
            epsilon = self.epsilon
        else:
            epsilon = self.epsilon

        for state in states:
            state = torch.from_numpy(state).float()
            q_value = self.eval_sub_net(state)
            if np.random.uniform() > epsilon:
                action = torch.argmax(q_value)
            else:
                action = np.random.choice(self.n_actions)
            actions.append(action)

        if self.epsilon > 0.1:
            self.epsilon *= 0.9999
        return actions

    def store_transition(self, states, actions, reward, done, next_states):
        # print(len(states),len(actions), len(reward), len(done), len(next_states))
        states = np.array(states).reshape(-1)
        actions = np.array(actions)
        reward = np.array(reward)
        done = np.array(done)
        next_states = np.array(next_states).reshape(-1)
        # print(states.shape, actions.shape, reward.shape, done.shape, next_states.shape)
        transition = np.hstack((states, actions, reward, done, next_states))
        # replace the old memory with new memory
        index = self.memory_counter % self.conf.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.train_step += 1
        batch = self.sample_data()
        states, actions, rewards, dones, next_states = batch

        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(-1).transpose(0, 1)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        q_evals, q_targets = self.get_q_values(batch)

        try:
            q_evals = torch.gather(q_evals, dim=2, index=actions).squeeze()
        except:
            import pdb; pdb.set_trace()

        q_targets = q_targets.max(dim=2)[0]

        """
        if not q_evals.shape[0] == 32:
            import pdb
            pdb.set_trace()
        """
        q_total_eval = self.eval_mix_net(q_evals, states).squeeze(1)
        q_total_target = self.target_mix_net(q_targets, next_states).squeeze(1)

        try:
            targets = rewards + self.conf.GAMMA * q_total_target * (1 - dones)
        except:
            import pdb
            pdb.set_trace()

        print("q_targets", targets)
        print("q_evals", q_total_eval)
        self.loss = self.loss_func(q_total_eval, targets.detach())
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if self.train_step > 0 and self.train_step % self.conf.update_target_params == 0:
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            self.target_sub_net.load_state_dict(self.eval_sub_net.state_dict())

    def get_q_values(self, batch):
        q_evals = []
        q_targets = []
        for data in range(self.conf.BATCH_SIZE):
            sub_state, sub_state_ = self.get_state(batch[0][data], batch[-1][data])
            sub_state = sub_state.to(self.device)
            sub_state_ = sub_state_.to(self.device)

            q_eval = self.eval_sub_net(sub_state)
            q_target = self.target_sub_net(sub_state_)
            q_eval = q_eval.view(self.conf.n_agents, -1)
            q_target = q_target.view(self.conf.n_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_target)

        # import pdb; pdb.set_trace()
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        return q_evals, q_targets

    @staticmethod
    def get_state(sub_state, sub_state_):
        sub_state = sub_state.reshape(8, 1156)
        sub_state_ = sub_state_.reshape(8, 1156)
        return sub_state, sub_state_

    def sample_data(self):
        sample_index = np.random.choice(self.conf.MEMORY_CAPACITY, self.conf.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.conf.N_STATES*8])
        b_a = torch.LongTensor(b_memory[:, self.conf.N_STATES*8:self.conf.N_STATES*8 + 8].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.conf.N_STATES*8 + 8:self.conf.N_STATES*8 + 9])
        b_done = torch.FloatTensor(b_memory[:, self.conf.N_STATES*8 + 9:self.conf.N_STATES*8 + 10])
        b_s_ = torch.FloatTensor(b_memory[:, - self.conf.N_STATES*8:])
        return b_s, b_a, b_r, b_done, b_s_

    def save(self):
        torch.save(self.eval_sub_net.state_dict(), "./model/q_mix_sub_net/subnet.pth")
