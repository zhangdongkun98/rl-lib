

from itertools import count

import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

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


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        try: nn.init.constant_(m.bias, 0.01)
        except: pass
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith('weight'): nn.init.orthogonal_(param)
    return

class ActorCritic(nn.Module):
    def __init__(self, dim_state, dim_action, device):
        super(ActorCritic, self).__init__()

        self.device = device

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
        state = torch.from_numpy(state).float().to(self.device) 
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
    weight_entropy = 0.001
    K_epochs = 4

    def __init__(self, config, device):
        dim_state, dim_action = config.dim_state, config.dim_action
        lr, betas = config.lr, eval(config.betas)
        self.gamma = config.gamma

        self.device = device

        ### ppo
        self.epsilon_clip = config.epsilon_clip
        # self.K_epochs = config.K_epochs
        self.wv, self.we = config.weight_value, self.weight_entropy
        
        self.policy = ActorCritic(dim_state, dim_action, device).to(device)
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

        # import pdb; pdb.set_trace()
        
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

            # import pdb; pdb.set_trace()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.wv*self.MseLoss(state_values, rewards) - self.we*dist_entropy
            loss = loss.mean()
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            writer.add_scalar('loss/loss', loss.detach().item(), self.train_step)
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='None', help='[Method] description.')

    args = argparser.parse_args()
    return args



def main():
    utils.setup_seed(1998)

    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"

    method_name = 'PPO'

    config = utils.parse_yaml_file_unsafe('./param.yaml')
    args = generate_args()
    config.update(args)
    writer = utils.create_dir(config, method_name + '_' + env_name)
    # path_pack = writer.path_pack
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    env = gym.make(env_name)
    config.set('dim_state', env.observation_space.shape[0])
    config.set('dim_action', 4)

    solved_reward = 230         # stop training if avg_reward > solved_reward
    update_timestep = 2000      # update policy every n timesteps



    #############################################
    
    ppo = PPO(config, device)
    
    timestep = 0
    for i_episode in range(1, config.max_episodes+1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        t = 0
        for t in range(config.max_timesteps):
        # for t in count():
            timestep += 1
            
            # Running policy_old:
            action, logprobs = ppo.policy_old.act(state)


            next_state, reward, done, _ = env.step(action.item())

            # Saving reward and is_terminal:
            ppo.memory.states.append(torch.from_numpy(state).float().to(device))
            ppo.memory.actions.append(action)
            ppo.memory.logprobs.append(logprobs)
            ppo.memory.rewards.append(reward)
            ppo.memory.is_terminals.append(done)

            state = next_state

            # update if its time
            if timestep % update_timestep == 0:
                print('memory size: ', len(ppo.memory.states), len(ppo.memory.actions), len(ppo.memory.rewards), len(ppo.memory.logprobs), len(ppo.memory.is_terminals))
                ppo.update(writer)
                timestep = 0
            
            running_reward += reward
            if config.render: env.render()
            if done: break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            
        # logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)
            
if __name__ == '__main__':
    main()
    
