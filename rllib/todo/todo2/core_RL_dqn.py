import intersection_env
import dqn
import torch
import numpy as np

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

env = intersection_env.IntersectionEnv()

def main():
    env_name = "intersection-v0"
    # creating environment
    dqn_policy = dqn.DQN()
    RENDER = True
    print('\nCollecting experience...')
    for i_episode in range(20000):
        s = env.reset()
        s = np.reshape(s, -1)
        ep_r = 0
        while True:
            if RENDER:
                env.env.render()
            a = dqn_policy.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            s_ = np.reshape(s_, -1)
            dqn_policy.store_transition(s, a, r, 1-done, s_)

            ep_r += r
            if dqn_policy.memory_counter > MEMORY_CAPACITY:
                dqn_policy.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2), '|epsilon:', dqn_policy.epsilon)

            if done:
                break

            if i_episode % 100 == 0:
                RENDER = True
            else:
                RENDER = True

            s = s_
    dqn_policy.save()

if __name__ == '__main__':
    main()
