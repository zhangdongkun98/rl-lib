import gym
import highway_env


class HighwayEnv(object):
    def __init__(self):
        self.env = gym.make("highway-v0")
        config = self.env.default_config()
        self.env.HIGH_SPEED_REWARD = 0.2
        self.env.ARRIVED_REWARD = 5
        print("default_config:", config)
        observation = {
            "type": "OccupancyGrid",
            "vehicles_count": 40,
            "features": ['presence', 'vx', 'vy'],
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
        duration = 40
        config['observation'] = observation
        config['vehicles_count'] = 40
        config['duration'] = duration
        config['show_trajectories'] = True

        self.env.configure(config)
        print(config)

        self.state = None
        self.reward = 0
        self.done = 0

    def reset(self):
        """
        :return: Reset the env and return the state observation.
        """

        state_ = self.env.reset()
        return state_

    def step(self, action):
        """
        :param action: From core_RL as action input.
        :return: next state, reward, done and info as return.
        """
        state_, reward, done, info = self.env.step(action)

        state_, reward, done, info = self.perception(state_, reward, done, info)
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

