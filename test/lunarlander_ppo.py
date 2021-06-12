import carla_utils as cu
import rllib

import torch

from rllib.args import generate_args
from rllib.gallery import init_ppo


def main():
    seed = 1998
    rllib.utils.setup_seed(seed)

    ############## Hyperparameters ##############

    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    max_episodes = 10000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    update_timestep = 2000      # update policy every n timesteps
    
    config = cu.system.YamlConfig({}, 'None')
    args = generate_args()
    config.update(args)  

    writer, env, ppo = init_ppo(config, seed)

    #############################################

    for i_episode in range(1, max_episodes+1):
        running_reward = 0
        avg_length = 0
        state = env.reset()
        for _ in range(max_timesteps):
            action = ppo.select_action( torch.from_numpy(state).unsqueeze(0).float() )

            next_state, reward, done, _ = env.step(action.cpu().numpy().flatten())

            experience = cu.rl_template.Experience(
                    state=torch.from_numpy(state).float().unsqueeze(0).to(config.device),
                    action=action, reward=reward, done=done)   ## change into CPU
            ppo.store(experience)

            state = next_state

            ppo.update_policy()
            
            running_reward += reward
            avg_length += 1
            if render: env.render()
            if done: break
        
        # stop training if avg_reward > solved_reward
        if running_reward > solved_reward:
            print("########## Solved! ##########")
            
        # logging
        print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
        writer.add_scalar('index/reward', running_reward, i_episode)
        writer.add_scalar('index/avg_length', avg_length, i_episode)
            
if __name__ == '__main__':
    main()
    
