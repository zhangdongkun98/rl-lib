import carla_utils as cu
import carla

import numpy as np
import multiprocessing as mp
import time
import copy
import random
import os, sys
from os.path import join, dirname

from config.sensor import sensors_param_list
from env.agents_master import AgentPseudo, AgentReal


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--use-kb-control', action='store_true', help='Haha')
    argparser.add_argument('--debug', action='store_true', help='zeze')
    argparser.add_argument('-v', '--max-velocity', metavar='P', default=8.34, type=float, help='gaga')
    argparser.add_argument('-d', '--decision-frequency', metavar='P', default=3, type=int, help='gaga')
    argparser.add_argument('-c', '--control-frequency', metavar='P', default=39, type=int, help='gaga')
    argparser.add_argument('--fast', action='store_true', help='zeze')
    argparser.add_argument('--pseudo', action='store_true', help='zeze')
    args = argparser.parse_args()
    return args


class EgoVehicle(object):
    def __init__(self):
        '''parameter'''
        config = cu.parse_yaml_file_unsafe('./config/carla.yaml')
        args = generate_args()
        config.update(args)

        self.frequency = config.decision_frequency
        self.clock = cu.system.Clock(self.frequency)
        host = config.host
        port = config.port
        timeout = config.timeout
        map_name = config.map_name
        role_name = config.role_name
        type_id = config.type_id

        self.pseudo, self.fast = config.pseudo, config.fast
        self.settings = carla.WorldSettings()
        self.settings.synchronous_mode = self.pseudo
        self.settings.no_rendering_mode = False
        self.settings.fixed_delta_seconds = 0.005 if self.pseudo else 0.0

        client, world, town_map = cu.connect_to_server(host, port, timeout, map_name, settings=self.settings)
        self.client, self.world, self.town_map = client, world, town_map
        
        start_point = carla.Location(x=100, y=1.8)
        start_point = carla.Location(x=-2, y=20)
        straight = carla.Location(x=100, y=1.8)   # straight
        right = carla.Location(x=-1.8, y=100)   # right
        left = carla.Location(x=1.8, y=-100)   # left
        end_point = random.choice([straight, right, left])
        end_point = [straight, right, left][0]
        self.route_planner = cu.AgentsRoutePlanner(self.world, self.town_map, config)
        self.global_path = self.route_planner.trace_route(start_point, end_point)
        self.waypoints = self.global_path.carla_waypoints
        # cu.draw_waypoints(self.world, self.waypoints, size=0.1, color=(0,0,255), life_time=20)

        self.vehicle = cu.add_vehicle(world, town_map, start_point, type_id, role_name='0')

        self.sensors_master = cu.createSensorListMaster(client, world, self.vehicle, sensors_param_list)
        self.pygame_interaction = cu.PyGameInteraction(client, self.vehicle, self.sensors_master, config)

        # Agent = cu.BaseAgentPseudo if self.pseudo else cu.BaseAgent
        # self.base_agent = Agent(config, client, world, town_map, self.vehicle, self.sensors_master)

        Agent = AgentPseudo if self.pseudo else AgentReal
        self.base_agent = Agent(config, client, world, town_map, self.vehicle, self.sensors_master, self.global_path, 0)
        self.dist = 0.0


    def destroy(self):
        self.sensors_master.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None


    def run_step(self):
        """
            Note: needs world.tick() in this method.
        
        Args:
        ---------
        
        Returns:
        -------
        """

        a = self.vehicle.get_location()


        action = 2 if self.base_agent.tick_time < 20 else 0
        print(self.base_agent.tick_time)
        ac = self.vehicle.get_acceleration()
        print(self.base_agent.get_current_v(), np.sqrt(ac.x**2 + ac.y**2 + ac.z**2))
        
        self.base_agent.run_step(action)
        if self.settings.synchronous_mode: self.world.tick()

        b = self.vehicle.get_location()
        # print(a.distance(b))

        if self.base_agent.tick_time >= 20:
            self.dist += a.distance(b)
            print(self.dist)

        self.base_agent.tick_time += 1



    def run(self):
        while True:
            t1 = self.clock.tick_begin()
            self.run_step()
            self.pygame_interaction.tick()

            if not self.fast: t2 = self.clock.tick_end()
            else: t2 = time.time()
            # print('freq: ', 1/(t2-t1))
            print()
    


if __name__ == '__main__':
    ego_vehicle = None
    try:
        ego_vehicle = EgoVehicle()
        ego_vehicle.run()
        pass
    except KeyboardInterrupt:
        print('canceled by user')
    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        print('destroyed all relevant actors')
