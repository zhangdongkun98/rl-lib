import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import copy

import utils

from methods.tools import init_weights



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
    def __init__(self, config, std_action):
        super(ActorCritic, self).__init__()

        self.device = config.device

        self.actor = nn.Sequential(
            nn.Linear(config.dim_state, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, config.dim_action), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.dim_state, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.apply(init_weights)
        self.var = torch.full((config.dim_action,), std_action**2).to(self.device)
        self.cov = torch.diag(self.var)
        
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, self.cov)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        var = self.var.expand_as(action_mean)
        cov = torch.diag_embed(var).to(self.device)

        dist = MultivariateNormal(action_mean, cov)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)


        # import pdb; pdb.set_trace()

        return action_logprobs, state_value.squeeze(), dist_entropy




class PPO(object):
    weight_value = 1.0
    weight_entropy = 0.0
    K_epochs = 4

    epsilon_clip = 0.2
    betas = (0.9, 0.999)

    lr = 0.002
    gamma = 0.99

    std_action = 0.5

    def __init__(self, config):

        self.device = config.device
        
        self.policy = ActorCritic(config, self.std_action).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        
        self.MseLoss = nn.MSELoss()
        self.MseLoss = nn.MSELoss(reduction='none')
        self.memory = Memory()

        self.train_step = -1
    
    
    def update(self, writer):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach().squeeze()

        for _ in range(self.K_epochs):
            self.train_step += 1
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages

            surr2 = torch.clamp(ratios, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.weight_value*self.MseLoss(state_values, rewards) - self.weight_entropy*dist_entropy
            
            import pdb; pdb.set_trace()
            
            
            loss = loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            writer.add_scalar('loss/loss', loss.detach().item(), self.train_step)
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()
        
    
    def select_action(self, state):
        action, log_prob = self.policy_old.act(state.to(self.device))
        return action, log_prob





from utils import generate_args

def main():
    seed = 1998
    utils.setup_seed(seed)

    ############## Hyperparameters ##############
    env_name = "LunarLanderContinuous-v2"
    method_name = 'PPO'

    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    max_episodes = 10000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    update_timestep = 2000      # update policy every n timesteps
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = utils.parse_yaml_file_unsafe('./param.yaml')
    args = generate_args()
    config.update(args)

    env = gym.make(env_name)
    env.seed(seed)
    # env.action_space.seed(seed)
    config.set('dim_state', env.observation_space.shape[0])
    config.set('dim_action', env.action_space.shape[0])
    config.set('device', device)

    writer = utils.create_dir(config, method_name + '_' + env_name)

    #############################################


    ## notice demension
    
    ppo = PPO(config)

    # from methods.ppo_continuous import Actor, Critic
    # config.set('net_actor', Actor)
    # config.set('net_critic', Critic)
    # from methods import ppo_continuous
    # ppo = ppo_continuous.PPO(config, writer)
    
    timestep = 0
    for i_episode in range(1, max_episodes+1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        t = 0
        for t in range(max_timesteps):
        # for t in count():
            timestep += 1
            
            # Running policy_old:
            action, logprobs = ppo.select_action( torch.from_numpy(state).unsqueeze(0).float() )

            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            # import pdb; pdb.set_trace()

            # Saving reward and is_terminal:
            ppo.memory.states.append(torch.from_numpy(state).float().to(device))
            ppo.memory.actions.append(action.squeeze())
            ppo.memory.logprobs.append(logprobs)
            ppo.memory.rewards.append(reward)
            ppo.memory.is_terminals.append(done)


            # ppo._replay_buffer.states.append(torch.from_numpy(state).float().to(device))
            # ppo._replay_buffer.actions.append(action.squeeze())
            # ppo._replay_buffer.logprobs.append(logprobs)
            # ppo._replay_buffer.rewards.append(reward)
            # ppo._replay_buffer.dones.append(done)

            state = next_state

            # update if its time
            if timestep % update_timestep == 0:
                # print('memory size: ', len(ppo.memory.states), len(ppo.memory.actions), len(ppo.memory.rewards), len(ppo.memory.logprobs), len(ppo.memory.is_terminals))
                ppo.update(writer)
                timestep = 0
            
            running_reward += reward
            if render: env.render()
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
    
