import carla_utils as cu

from itertools import count
import sys
sys.path.append('/home/ff/github/zju/intersection_learning')

import gym
import torch

from method_ddpg.ddpg import DDPG as Method

from utils import tools, Experience, States
tools.setup_seed(9527)

config = cu.parse_yaml_file_unsafe('./config/carla.yaml')
param = cu.parse_yaml_file_unsafe('./todo/param.yaml')
config.update(param)
from config.args import generate_args
args = generate_args()
config.update(args)

path_pack, logger = cu.basic.create_dir(config, 'ddpg_lunar')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### pip install box2d-py



def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    render = False
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 1  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    #############################################

    method = Method(device, config, path_pack)
    method.train()

    indices = torch.tensor([0])
    masks = torch.tensor([1])

    # logging variables
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        for _ in count():
            timestep += 1
            avg_length += 1
            # Running policy_old:
            action, action_prob = method.select_action(torch.tensor(state))
            next_state, reward, done, _ = env.step(action)

            # action_prob = torch.tensor(cu.basic.int2onehot(action, config.dim_action), dtype=torch.float32)
            experience = Experience(States(torch.tensor(state, dtype=torch.float32), indices.clone()),
                    action_prob, States(torch.tensor(next_state, dtype=torch.float32), indices.clone()),
                    torch.tensor([reward], dtype=torch.float32), torch.tensor([done], dtype=torch.float32), masks.clone())
            method.memory.push(experience)

            state = next_state
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        for _ in range(200):
            c_loss, a_loss = method.update_policy()
            logger.add_scalar('loss/c_loss', c_loss, method.train_step)
            logger.add_scalar('loss/a_loss', a_loss, method.train_step)



        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")

        # logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        logger.add_scalar('reward/epr', running_reward, i_episode)
        logger.add_scalar('param/e_greedy', method.epsilon_prob, i_episode)


        

if __name__ == '__main__':
    main()
