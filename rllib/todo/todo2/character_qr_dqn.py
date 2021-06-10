from torch.utils.tensorboard import SummaryWriter

import os
from os.path import join
import numpy as np
import time
import random

import carla_character_multi_direction
import qr_dqn_independent

myenv = carla_character_multi_direction.CarlaEnv()
mypolicy = qr_dqn_independent.QR_DQN()
private_policy_list = []
for i in range(4):
    sub_policy = qr_dqn_independent.QR_DQN()
    private_policy_list.append(sub_policy)

character_dict = dict({0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3})
test_episode = 14000
MEMORY_CAPACITY = 10000

'''params'''
description = 'zdk-learn'
dataset_name = 'qr_dqn/' + str(int(time.time())) + ' -- ' + description
log_path = join('results', dataset_name, 'log')
save_model_path = join('results', dataset_name, 'saved_models')
output_path = join('results', dataset_name, 'output')
os.makedirs(save_model_path)
os.makedirs(output_path)
logger = SummaryWriter(log_dir=log_path)

steps_done = 0
running_reward = None
gamma, batch_size = 0.95, 32

policy = qr_dqn_independent.QR_DQN()
thirty_rewards = [-1000 for i in range(30)]
best_avg_reward = -1000
LEARN = True
alpha = 1

try:
    print("trying")
    reward_list = []
    for i in range(test_episode):
        """
        for policy_index, policy in enumerate(private_policy_list):
            policy_name = "./model/0709_test/multi_iql_net-1.pth"
            policy.restore(policy_name, 0.05)
        """
        states, characters = myenv.reset()
        ep_r = 0
        step_index = 0
        while True:
            actions = []
            for j in range(len(states)):
                public_action = mypolicy.choose_action(np.array(states[j]))
                private_action = private_policy_list[character_dict[characters[j]]].choose_action(states[j])
                randn = random.random()
                if randn < alpha:
                    actions.append(public_action)
                else:
                    # print('here')
                    actions.append(private_action)

            # print("actions:", actions)
            states_, rewards, dones, characters = myenv.step(actions)
            for k in range(len(states)):
                mypolicy.store_transition(states[k], actions[k], states_[k], rewards[k], dones[k])
                private_policy_list[character_dict[characters[k]]].store_transition(
                    states[k], actions[k], states_[k], rewards[k], dones[k]
                )
            if mypolicy.memory_counter > MEMORY_CAPACITY and LEARN:
                # print('learning')
                mypolicy.learn()
                for policy in private_policy_list:
                    # print('private')
                    policy.learn()

            states = []

            for index, done_info in enumerate(dones):
                if done_info != 1:
                    states.append(states_[index])

            ep_r += sum(rewards)

            if sum(dones) == len(dones):
                break
            step_index += 1
            alpha *= 0.99999

        logger.add_scalar('reward/epr', ep_r, i)
        logger.add_scalar('training/epsilon', policy.eps, i)
        logger.add_scalar('training/alpha', alpha, i)
        logger.add_scalar('training/q_loss', policy.loss, i)
        logger.add_scalar('training/iter_steps', step_index, i)
        print('Ep: ', i, '| Ep_r: ', round(ep_r, 2), '|epsilon', policy.eps, "iter_steps", step_index, "alpha", alpha)
        print('\n\n\n')
        reward_list.append(ep_r)
        thirty_rewards[i % 30] = ep_r
        tmp_avg_reward = sum(thirty_rewards) / 30.0
        if tmp_avg_reward > best_avg_reward:
            mypolicy.save(-1000, save_model_path)
            for nn, policy in enumerate(private_policy_list):
                policy.save(nn+1000, save_model_path)
            best_avg_reward = tmp_avg_reward
finally:
    myenv.destroy(myenv.player_list)
    print(reward_list)
    mypolicy.save(-1, save_model_path)
    for nn, policy in enumerate(private_policy_list):
        policy.save(nn, save_model_path)
