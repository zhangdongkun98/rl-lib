import carla_env
import independent_dqn
import numpy as np

myenv = carla_env.CarlaEnv()
mypolicy = independent_dqn.DQN()

test_episode = 1400
MEMORY_CAPACITY = 3000
with open("offline-dqn_data-0618.txt", 'w') as f:
    try:
        #mypolicy.restore("./model/eval_net_288_offline_deeptrain.pth", 0.95)
        mypolicy.restore("./model/eval_net_highway_env.pth", 0.95)
        reward_list = []
        for i in range(test_episode):
            states = myenv.reset()
            ep_r = 0
            while True:
                actions = []
                for j in range(len(states)):
                    action = mypolicy.choose_action(np.array(states[j]))
                    actions.append(action)
                #print(actions)
                states_, rewards, dones = myenv.step(actions)

                for k in range(len(states)):
                    mypolicy.store_transition(states[k], actions[k], rewards[k],  dones[k], states_[k])
                    f.write('!')
                    f.write(str(states[k]))
                    f.write('@')
                    f.write(str(actions[k]))
                    f.write('#')
                    f.write(str(rewards[k]))
                    f.write('$')
                    f.write(str(dones[k]))
                    f.write('%')
                    f.write(str(states_[k]))
                    f.write('\n')

                if mypolicy.memory_counter > MEMORY_CAPACITY:
                    mypolicy.learn()

                states = []
                remove_list = []

                for index, done_info in enumerate(dones):
                    if done_info != 1:
                        states.append(states_[index])

                ep_r += sum(rewards)

                if sum(dones) == len(dones):
                    break

            print('Ep: ', i, '| Ep_r: ', round(ep_r, 2), '|epsilon', mypolicy.epsilon)
            reward_list.append(ep_r)

    finally:
        myenv.destroy(myenv.player_list)
        mypolicy.save(1)