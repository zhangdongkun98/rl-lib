import carla_env
import carla_policy
import numpy as np


myenv = carla_env.CarlaEnv()
mypolicy = carla_policy.Policy()

test_episode = 100
try:
    for i in range(test_episode):
        states = myenv.reset()
        while True:
            actions = mypolicy.choose_action(np.array(states))
            states_, rewards, dones = myenv.step(actions)
            print(dones)
            #print(actions)
            if sum(dones) == len(dones):
                break

            states = states_
finally:
    myenv.destroy(myenv.player_list)