import intersection_env
import qr_dqn_independent
import torch
import numpy as np


def main():
    ############## Hyperparameters ##############
    env_name = "intersection-v0"
    # creating environment
    env = intersection_env.IntersectionEnv()
    render = True
    log_interval = 20  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 500  # max timesteps in one episode

    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    mypolicy = qr_dqn_independent.QR_DQN()
    mypolicy.restore("./model/character_0623-2/multi_iql_net1000.pth", 0.05)

    # logging variables

    timestep = 0
    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        state = np.reshape(state, -1)
        running_reward = 0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = mypolicy.choose_action(state)
            state, reward, done, _ = env.step(action)
            print(action, reward, done)
            state = np.reshape(state, -1)
            reward = float(reward)
            # Saving reward and is_terminal:

            # update if its time

            running_reward += reward
            if render:
                env.env.render()
            if done:
                print("episode_index:", i_episode, "running_reward:", running_reward)
                break

        if i_episode % 100 == 0:
            render = True
        else:
            render = True

if __name__ == '__main__':
    main()
