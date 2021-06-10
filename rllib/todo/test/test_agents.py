import carla_utils as cu
import carla

import numpy as np
import time
import copy
import random
import os, sys
from os.path import join, dirname

from config.sensor import sensors_param_list


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

        nv = 1
        spawn_points = cu.get_spawn_points(town_map, nv)
        vehicles = cu.add_vehicles(client, world, town_map, spawn_points, [type_id]*len(spawn_points), role_names=[role_name]*len(spawn_points))
        sensors_masters = cu.createSensorListMasters(client, world, vehicles, [sensors_param_list]*len(vehicles))

        # Agent = cu.NaiveAgentPseudo if self.pseudo else cu.NaiveAgent
        Agent = cu.BaseAgentPseudo if self.pseudo else cu.BaseAgent
        self.agents_master = cu.AgentListMaster(config, client, world)
        for vehicle, sensors_master in zip(vehicles, sensors_masters):
            agent = Agent(config, client, world, town_map, vehicle, sensors_master)
            self.agents_master.register(agent)

        self.route_planner = cu.AgentsRoutePlanner(self.world, self.town_map, config)

        # self.pygame_interaction = cu.PyGameInteraction(client, vehicles[0], sensors_masters[0], config)



    def destroy(self):
        self.agents_master.destroy()


    def run_step(self):
        """
            Note: needs world.tick() in this method.
        
        Args:
        ---------
        
        Returns:
        -------
        """
        
        self.agents_master.run_step()
        if self.settings.synchronous_mode: self.world.tick()


    def run(self):
        while True:
            self.clock.tick_begin()
            self.run_step()
            # self.pygame_interaction.tick()

            if not self.fast: self.clock.tick_end()
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
