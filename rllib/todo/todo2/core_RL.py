import intersection_env
import ppo
import torch
import numpy as np


def main():
    ############## Hyperparameters ##############
    env_name = "intersection-v0"
    # creating environment
    env = intersection_env.IntersectionEnv()
    state_dim = 900
    action_dim = 3
    render = True
    solved_reward = 15 # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 30000  # max training episodes
    max_timesteps = 500  # max timesteps in one episode
    n_latent_var = 288  # number of variables in hidden layer
    update_timestep = 80  # update policy every n timesteps
    lr = 0.0002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    f = open("./reward.txt", 'w')
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = ppo.Memory()
    myppo = ppo.PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    running_reward_list = []
    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        state = np.reshape(state, -1)
        #print(state)
        #state = np.expand_dims(state, 0)
        #print(state)

        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action = myppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            print(action, reward, done)
            #state = np.expand_dims(state, 0)
            state = np.reshape(state, -1)
            reward = float(reward)
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                myppo.update(memory)
                torch.cuda.empty_cache()
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if i_episode % 10000 == 0:
            print("########## Solved! ##########")
            torch.save(myppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))

        # logging
        if i_episode % 100 == 0:
            render = True
        else:
            render = False

        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = (running_reward / log_interval)

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            f.write(str(running_reward) + "\n")
            running_reward_list.append(running_reward)
            running_reward = 0
            avg_length = 0

        if i_episode % 1000 == 0:
            print(running_reward_list)

if __name__ == '__main__':
    main()
