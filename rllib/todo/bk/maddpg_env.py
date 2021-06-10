import carla_utils as cu
from carla_utils import carla
ApplyTransform = carla.command.ApplyTransform
DestroyActor = carla.command.DestroyActor

import numpy as np

from utils import env_multi_agent


class CarlaEnv(env_multi_agent.CarlaEnv):
    def _reset_done_vehicles(self, dones):
        batch, done_agents = [], []
        for agent, done in zip(self.agents_master.agents_learnable, dones):
            if done:
                done_agents.append(agent)
                batch.extend(agent.destroy_commands())
        for done_agent in done_agents: self.agents_master.remove(done_agent)
        self.client.apply_batch_sync(batch)
        cu.tick_world(self.world)

    def _calculate_reward(self, collisions, timeouts, successes):
        rewards = []
        for (agent, collision, timeout, _) in zip(self.agents_master.agents_learnable, collisions, timeouts, successes):
            state = agent.get_state()

            reward_dense = (state.v - self.max_velocity / 2) * 0.025
            reward_collision = int(collision) * self.reward_collision

            distance_to_intersection = -(state.x * np.cos(state.theta) + state.y * np.sin(state.theta))
            reward_done = int(timeout) * (distance_to_intersection < -self.intersection_threshold)

            reward = reward_dense + reward_collision + reward_done
            rewards.append(reward)
        return rewards
