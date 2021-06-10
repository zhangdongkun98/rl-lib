import gym
import highway_env
import numpy as np


class IntersectionEnv(object):
    def __init__(self):
        self.env = gym.make("intersection-v0")
        config = self.env.default_config()
        self.env.HIGH_SPEED_REWARD = 0.2
        self.env.ARRIVED_REWARD = 5
        observation = {
            "type": "OccupancyGrid",
            "vehicles_count": 15,
            "features": ['presence', 'vy'],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-30, 30], [-30, 30]],
            "grid_step": [5, 5],
            "absolute": False
        }
        duration = 25
        config['observation'] = observation
        config['duration'] = duration
        config['show_trajectories'] = True
        print(config)
        self.env.configure(config)

        self.state = None
        self.reward = 0
        self.done = 0

    def reset(self):
        """
        :return: Reset the env and return the state observation.
        """

        state_ = self.env.reset()
        state_[0][6][6] = 0.0
        state_ = np.append(state_, 0.5)
        return state_

    def step(self, action):
        """
        :param action: From core_RL as action input.
        :return: next state, reward, done and info as return.
        """
        state_, reward, done, info = self.env.step(action)
        state_[0][6][6] = 0.0
        print(state_)
        state_ = np.append(state_, 0.5)
        arrive = self.has_arrived
        if arrive:
            reward += 5
            done = 1
        state_, reward, done, info = self.perception(state_, reward, done, info)
        #cv2.imshow("state", state_)
        #cv2.waitKey(1)
        # print(self.env.vehicle.lane_index)
        # print(self.env.vehicle.lane.local_coordinates(self.env.vehicle.position), self.env.vehicle.lane.length - 15 * self.env.vehicle.LENGTH)
        # print(self.env.HIGH_SPEED_REWARD, self.env.vehicle.speed_index, self.env.vehicle.SPEED_COUNT)
        return state_, reward, done, info

    @staticmethod
    def perception(state_, reward, done, info):
        return state_, reward, done, info

    @staticmethod
    def cal_reward():
        return 0

    @property
    def has_arrived(self):
        return "il" in self.env.vehicle.lane_index[0] \
               and "o" in self.env.vehicle.lane_index[1] \
               and self.env.vehicle.lane.local_coordinates(self.env.vehicle.position)[0] >= \
               self.env.vehicle.lane.length - 15 * self.env.vehicle.LENGTH