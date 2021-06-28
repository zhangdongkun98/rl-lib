import rllib

import gym
import torch

from rllib.args import generate_args


def main():
    seed = 1998
    rllib.basic.setup_seed(seed)

    ############## Hyperparameters ##############

    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    max_episodes = 10000        # max training episodes
    
    config = rllib.basic.YamlConfig({}, 'None')
    args = generate_args()
    config.update(args)

    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    setattr(env, 'dim_state', env.observation_space.shape[0])
    setattr(env, 'dim_action', env.action_space.shape[0])

    from rllib.ddpg import DDPG as Method

    config.set('dim_state', env.dim_state)
    config.set('dim_action', env.dim_action)
    config.set('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    model_name = Method.__name__ + '-' + env_name
    writer = rllib.basic.create_dir(config, model_name)
    method = Method(config, writer)

    #############################################

    for i_episode in range(max_episodes):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        while True:
            action = method.select_action( torch.from_numpy(state).unsqueeze(0).float() )
            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            experience = rllib.template.Experience(
                    state=torch.from_numpy(state).float().unsqueeze(0),
                    next_state=torch.from_numpy(next_state).float().unsqueeze(0),
                    action=action.cpu(), reward=reward, done=done)
            method.store(experience)

            state = next_state

            method.update_policy()

            running_reward += reward
            avg_length += 1
            if render: env.render()
            if done: break
        
        ### stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            
        ### logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)


            
if __name__ == '__main__':
    main()
    
