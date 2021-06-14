import carla_utils as cu
import rllib

import gym
import torch

from rllib.args import generate_args


def main():
    seed = 1998
    rllib.utils.setup_seed(seed)

    ############## Hyperparameters ##############

    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    max_episodes = 1000000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    update_timestep = 2000      # update policy every n timesteps
    
    config = cu.system.YamlConfig({}, 'None')
    args = generate_args()
    config.update(args)  

    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)
    env.seed(seed)
    setattr(env, 'dim_state', env.observation_space.shape[0])
    setattr(env, 'dim_action', env.action_space.shape[0])

    from rllib.td3 import TD3

    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    config.set('net_actor', rllib.td3.Actor)
    config.set('net_critic', rllib.td3.Critic)

    model_name = TD3.__name__ + '-' + env_name
    writer = cu.basic.create_dir(config, model_name)
    td3 = TD3(config, writer)

    #############################################

    for i_episode in range(1, max_episodes+1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        for _ in range(max_timesteps):
            action = td3.select_action( torch.from_numpy(state).unsqueeze(0).float() )

            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            experience = cu.rl_template.Experience(
                    state=torch.from_numpy(state).float().unsqueeze(0),
                    next_state=torch.from_numpy(next_state).float().unsqueeze(0),
                    action=action.cpu(), reward=reward, done=done)
            td3.store(experience)

            # import pdb; pdb.set_trace()

            state = next_state

            running_reward += reward
            avg_length += 1
            if render: env.render()
            if done: break
        
        td3.update_policy()

        # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            
        # logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)
            
if __name__ == '__main__':
    main()
    
