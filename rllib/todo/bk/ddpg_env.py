import carla_utils as cu
from carla_utils import carla
ApplyTransform = carla.command.ApplyTransform
DestroyActor = carla.command.DestroyActor

import numpy as np

from utils import env_single_agent


class CarlaEnv(env_single_agent.CarlaEnv):
    def _reset_done_vehicles(self, *args):
        return

    def _calculate_reward(self, agent, collision, timeout, success):
        state = agent.get_state()

        reward_dense = (state.v - self.max_velocity / 2) * 0.025
        reward_collision = int(collision) * self.reward_collision

        distance_to_intersection = -(state.x * np.cos(state.theta) + state.y * np.sin(state.theta))
        reward_done = int(timeout) * (distance_to_intersection < -self.intersection_threshold)
        # reward_done = 0

        reward = reward_dense + reward_collision + reward_done
        return reward

