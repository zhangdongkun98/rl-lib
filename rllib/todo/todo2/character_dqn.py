from torch.utils.tensorboard import SummaryWriter

import os
from os.path import join
import time

import carla_character_multi_direction
import independent_dqn
import numpy as np

myenv = carla_character_multi_direction.CarlaEnv()
mypolicy = independent_dqn.DQN()
private_policy_list = []
for i in range(4):
    sub_policy = independent_dqn.DQN()
    private_policy_list.append(sub_policy)

character_dict = dict({0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3})
test_episode = 3000
MEMORY_CAPACITY = 3000

description = 'zdk-learn'
dataset_name = 'qr_dqn/' + str(int(time.time())) + ' -- ' + description
log_path = join('results', dataset_name, 'log')
save_model_path = join('results', dataset_name, 'saved_models')
output_path = join('results', dataset_name, 'output')
os.makedirs(save_model_path)
os.makedirs(output_path)
logger = SummaryWriter(log_dir=log_path)

with open("offline-dqn_data-0621-1.txt", 'w') as f:
    try:
        #mypolicy.restore("./model/multi_iql_net_465iters.pth", 0.95)
        #mypolicy.restore("./model/character_0619-2/multi_iql_net-1.pth", 0.95)
        #for policy_index, policy in enumerate(private_policy_list):
            #if policy_index == 1:
            #    policy_name = "./model/character_0619-3/multi_iql_net-1.pth"
            #else:
            #    policy_name = "./model/character_0619-3/multi_iql_net" + str(3-policy_index) + ".pth"
            #policy_name = "./model/character_0620-1/multi_iql_net" + str(policy_index) + ".pth"

            #policy.restore(policy_name, 0.7)

        reward_list = []
        for i in range(test_episode):
            states, characters = myenv.reset()
            ep_r = 0
            step_index = 0
            while True:
                actions = []
                for j in range(len(states)):
                    public_action = mypolicy.choose_action(np.array(states[j]))
                    private_action = private_policy_list[character_dict[characters[j]]].choose_action(states[j])
                    actions.append(private_action)
                #print(actions)
                states_, rewards, dones, characters = myenv.step(actions)

                for k in range(len(states)):
                    mypolicy.store_transition(states[k], actions[k], rewards[k],  dones[k], states_[k])
                    private_policy_list[character_dict[characters[k]]].store_transition(
                        states[k], actions[k], rewards[k], dones[k], states_[k]
                    )

                if mypolicy.memory_counter > MEMORY_CAPACITY:
                    mypolicy.learn()
                    for policy in private_policy_list:
                        policy.learn()

                states = []

                for index, done_info in enumerate(dones):
                    if done_info != 1:
                        states.append(states_[index])

                ep_r += sum(rewards)

                if sum(dones) == len(dones):
                    break
                step_index += 1

            logger.add_scalar('reward/epr', ep_r, i)
            logger.add_scalar('training/epsilon', private_policy_list[0].epsilon, i)
            logger.add_scalar('training/q_loss', private_policy_list[0].loss, i)
            logger.add_scalar('training/iter_steps', step_index, i)
            print('Ep: ', i, '| Ep_r: ', round(ep_r, 2), '|epsilon', private_policy_list[0].epsilon)
            reward_list.append(ep_r)
    finally:
        myenv.destroy(myenv.player_list)
        print(reward_list)
        mypolicy.save(-1)
        for nn, policy in enumerate(private_policy_list):
            policy.save(nn)
