from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
# from smac.env import StarCraft2Env
import carla_character_multi_direction
from os.path import join
import time

from config import Config
from policy import QMixPolicy

conf = Config()

myenv = carla_character_multi_direction.CarlaEnv()
policy = QMixPolicy()
description = 'zdk-learn'
dataset_name = 'qr_dqn/' + str(int(time.time())) + ' -- ' + description
log_path = join('results', dataset_name, 'log')
save_model_path = join('results', dataset_name, 'saved_models')
output_path = join('results', dataset_name, 'output')
os.makedirs(save_model_path)
os.makedirs(output_path)
logger = SummaryWriter(log_dir=log_path)


train_steps = 0
try:
    for episode in range(conf.train_episodes):
        states, _ = myenv.reset()
        ep_r = 0
        step_index = 0
        while True:
            if len(states) != 8:
                break
            actions = policy.choose_actions(states, eval=True)
            # print(actions)
            states_, rewards, dones, characters = myenv.step(actions)
            reward = [rewards[-1]]
            done = [dones[-1]]

            # print(sum(done), len(done))
            if sum(done) == len(done):
                policy.store_transition(states, actions, reward, done, states)
            else:
                policy.store_transition(states, actions, reward, done, states_)
            if policy.memory_counter > conf.MEMORY_CAPACITY:
                policy.learn()

            states = states_
            ep_r += rewards[-1]
            step_index += 1
            if sum(done) == len(done):
                logger.add_scalar('reward/epr', ep_r, episode)
                logger.add_scalar('training/epsilon', policy.epsilon, episode)
                logger.add_scalar('training/q_loss', policy.loss, episode)
                logger.add_scalar('training/iter_steps', step_index, episode)
                print("episode: ", episode, " reset!", "now_counter:", policy.memory_counter)
                break
finally:
    policy.save()
    import pdb; pdb.set_trace()
    myenv.destroy(myenv.player_list)