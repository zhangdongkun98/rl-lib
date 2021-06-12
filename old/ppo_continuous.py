import carla_utils as cu

import torch
import gym

import utils
from utils import generate_args


def main():
    seed = 1998
    seed = 300
    utils.setup_seed(seed)

    ############## Hyperparameters ##############
    env_name = "LunarLanderContinuous-v2"
    method_name = 'PPO'

    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    max_episodes = 10000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = utils.parse_yaml_file_unsafe('./param.yaml')
    args = generate_args()
    config.update(args)

    env = gym.make(env_name)
    env.seed(seed)
    # (env.action_space.seed(seed))
    config.set('dim_state', env.observation_space.shape[0])
    config.set('dim_action', env.action_space.shape[0])
    config.set('device', device)
    # config.set('method', method_name)

    from methods.ppo_continuous import PPO
    from methods.ppo_continuous import Actor, Critic
    config.set('net_actor', Actor)
    config.set('net_critic', Critic)

    writer = utils.create_dir(config, method_name + '_' + env_name)
    ppo = PPO(config, writer)
    
    for i_episode in range(1, max_episodes+1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        for _ in range(max_timesteps):
            action = ppo.select_action( torch.from_numpy(state).unsqueeze(0).float() )

            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            experience = cu.rl_template.Experience(
                    state=torch.from_numpy(state).float().unsqueeze(0).to(device),
                    action=action, reward=reward, done=done)   ## change into CPU
            ppo.store(experience)

            state = next_state

            # print('memory size: ', len(ppo._replay_buffer.states), len(ppo._replay_buffer.actions), len(ppo._replay_buffer.rewards), len(ppo._replay_buffer.logprobs), len(ppo._replay_buffer.dones))
            ppo.update_policy()

           
            
            running_reward += reward
            avg_length += 1
            if render: env.render()
            if done: break
        
        # print('memory size: ', len(ppo._replay_buffer.states), len(ppo._replay_buffer.actions), len(ppo._replay_buffer.rewards), len(ppo._replay_buffer.logprobs), len(ppo._replay_buffer.dones))
        # ppo.update_policy()
        
        # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)
            
if __name__ == '__main__':
    main()
    
