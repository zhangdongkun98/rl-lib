from __future__ import print_function
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import os, sys
import math
import time
import argparse
import random

import carla
import local_planner
import numpy as np

sys.setrecursionlimit(10000)


class Controller(object):
    def __init__(self):
        self.max_speed = 8
        self.localPlanner = local_planner.LocalPlanner()
        pass

    def global_plan(self, transform, waypoint, apply_v):
        now_location = transform.location
        now_rotation = transform.rotation

        theta_error = now_rotation.yaw - waypoint.transform.rotation.yaw

        theta_error = theta_error / 180 * math.pi
        if theta_error > math.pi:
            theta_error -= 2 * math.pi
        elif theta_error < -math.pi:
            theta_error += 2 * math.pi

        dist_error = self.get_dist_error(now_location, waypoint)
        k_s = 0.0001
        k_theta = 0.2
        k_e = 0.05

        v_ref = apply_v

        apply_w = v_ref * k_s * math.cos(theta_error) / (1 - k_s * dist_error) - (k_theta * abs(v_ref)) * theta_error \
                  + (k_e * v_ref * math.sin(theta_error) / (theta_error + 0.0001)) * dist_error
        apply_w = apply_w * 180 / math.pi
        return apply_v, apply_w

    @staticmethod
    def get_dist_error(location, waypoint):
        """
        :param location:
        :param waypoint:
        :return:dist_error
        """
        rol = math.sqrt(
            (location.x - waypoint.transform.location.x) ** 2 + (location.y - waypoint.transform.location.y) ** 2)
        theta = math.atan2(waypoint.transform.location.y - location.y, (waypoint.transform.location.x - location.x))
        reltheta = theta - waypoint.transform.rotation.yaw / 180 * math.pi
        dist_error = rol * math.sin(reltheta)

        return dist_error

    def local_plan(self, action, player, route):
        return

    def PIDController(self, player, v_set, w_set):
        now_velocity = math.sqrt(player.get_velocity().x ** 2 + player.get_velocity().y ** 2)

        v_error = v_set - now_velocity

        brake = 0
        base_point = 0
        kp_v = 1

        throttle = base_point + kp_v * v_error / self.max_speed

        if v_error < -2 or v_set == 0:
            brake = 0.5 - kp_v * v_error / self.max_speed

        steer = math.radians(w_set) * 2.87 / (now_velocity + 0.1)
        if steer > math.pi / 6:
            steer = math.pi / 6
        elif steer < -math.pi / 6:
            steer = -math.pi / 6

        steer = steer / (math.pi / 6)

        # print("v_error", v_error, "w_error", w_error)

        # print("throttle:", throttle, "brake", brake, "steer", steer)
        if throttle > 1:
            throttle = 1
        elif throttle < 0:
            throttle = 0
        if brake > 1:
            brake = 1
        elif brake < 0:
            brake = 0
        if steer > 1:
            steer = 1
        elif steer < -1:
            steer = -1

        return throttle, steer, brake

    @staticmethod
    def find_nearest_waypoint(vehicle, route):
        player = vehicle.player
        target_location = player.get_location()
        min_dist = 10000
        # print("route_length:", len(route), "player:", player)
        min_index = 0
        for index, waypoint in enumerate(route):
            # print("index", index, min_dist, waypoint[0].transform.location.x, waypoint[0].transform.location.y)
            now_dist = math.sqrt((waypoint[0].transform.location.x - target_location.x) ** 2 +
                                 (waypoint[0].transform.location.y - target_location.y) ** 2)
            if now_dist < min_dist:
                min_dist = now_dist
                min_index = index

        if min_index + 15 < len(route):
            return route[min_index + 15][0]
        else:
            return route[min_index][0]

    def control(self, action, vehicle, route, local_planner_map):
        player = vehicle.player
        control_out = carla.VehicleControl()
        if action == 0:
            if vehicle.set_v >= 0:
                vehicle.set_v -= 1
            if vehicle.set_v < 0:
                vehicle.set_v = 0
        elif action == 1:
            vehicle.set_v = vehicle.set_v
        else:
            if vehicle.set_v < 8:
                vehicle.set_v += 1
            if vehicle.set_v > 8:
                vehicle.set_v = 8

        route_point = self.find_nearest_waypoint(vehicle, route)

        v_set, w_set = self.global_plan(player.get_transform(), route_point, vehicle.set_v)

        ori_v_set, ori_w_set = v_set, w_set
        v_set, w_set, point, reward = self.localPlanner.plan(v_set, w_set, vehicle.player.get_transform(),
                                                             local_planner_map, route_point)
        reward = (reward * 0.1 - 0.06)
        if v_set != ori_v_set or w_set != ori_w_set:
            # print("origin", vehicle.character_value, ori_v_set, ori_w_set)
            # print("planned", v_set, w_set, reward)
            reward -= 2.5

        vehicle.set_v = v_set
        control_out.throttle, control_out.steer, control_out.brake = self.PIDController(player, v_set, w_set)
        return control_out, route_point, point, reward


class Vehicle(object):
    def __init__(self, player, start, end, route, collision_sensor, character_value, direction):
        self.player = player
        self.start = start
        self.end = end
        self.route = route
        self.collision_sensor = collision_sensor
        self.collision_sensor.listen(lambda event: self.collision_event(event))
        self.is_collision = False
        self.clock = 0
        self.dist = 80.0
        self.set_v = 5
        self.character_value = character_value
        self.last_local_reward = 0
        self.direction = direction
        self.is_arrive = 0

    def check_done(self):
        now_location = self.player.get_location()
        self.dist = math.sqrt((now_location.x - self.end.x) ** 2 + (now_location.y - self.end.y) ** 2)
        if self.dist < 3.0 or self.is_collision or now_location.z < -5 or self.clock > 5000 or self.is_arrive == 1:
            if self.dist < 3.0 or self.is_arrive == 1:
                self.is_arrive = 1
                return 0, 1, 0
            elif self.is_collision:
                return 1, 0, 0
            else:
                return 0, 0, 1
        else:
            return 0, 0, 0

    def collision_event(self, event):
        self.is_collision = True
        return event


class CarlaEnv(object):
    def __init__(self):
        args = self.gen_args().parse_args()
        self.actor_role_name = args.rolename
        self._actor_filter = args.filterv
        self.player_list = []
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.character_dict = dict({0.0: False, 0.25: False, 0.5: False, 0.75: False})

        self.controller = Controller()
        self.last_arrive_list = [0 for _ in range(8)]

        dao = GlobalRoutePlannerDAO(self.carla_map, 0.1)
        self.grp = GlobalRoutePlanner(dao)
        self.grp.setup()

    def reset(self):
        """
        :return: Reset the env and return the first state
        """
        if self.player_list:
            self.destroy(self.player_list)
        args = self.gen_args().parse_args()
        self.actor_role_name = args.rolename
        self._actor_filter = args.filterv
        self.player_list = []
        self.client = carla.Client(args.host, args.port)
        self.world = self.client.get_world()

        spawn_points = self.get_random_spawn_points(8)
        self.controller = Controller()
        self.last_arrive_list = [0 for _ in range(8)]

        character_list = []
        spawn_random_list = [0.0, 0.0, 0.25, 0.25, 0.50, 0.50, 0.75, 0.75]
        random.shuffle(spawn_random_list)

        for spawn_index, spawn_point in enumerate(spawn_points):
            # randomly choose a blueprint
            blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
            blueprint.set_attribute('role_name', self.actor_role_name)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'true')

            player = self.world.try_spawn_actor(blueprint, spawn_point)
            i = 0
            while not player:
                player = self.world.try_spawn_actor(blueprint, spawn_point)
                i += 1
                if i > 10:
                    break
            time.sleep(0.15)
            direction = random.randint(0, 2)
            if player is not None:
                start = player.get_location()
                # print(start)
                """
                if start.x > 0 and start.y > 0:
                    end.x = -start.y
                    end.y = -start.x
                elif start.x > 0 and start.y < 0:
                    end.x = start.y
                    end.y = start.x
                elif start.x < 0 and start.y > 0:
                    end.x = start.y
                    end.y = start.x
                else:
                    end.x = -start.y
                    end.y = -start.x
                """
                end = self.get_end_point(start, direction)

                # print(start.x, start.y, end.x, end.y)
                waypoints = self.grp.trace_route(start, end)

                blueprint_library = self.world.get_blueprint_library()
                colsensor_bp = blueprint_library.find("sensor.other.collision")
                spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
                colsensor = self.world.spawn_actor(colsensor_bp, spawn_point, attach_to=player)

                character_value = spawn_random_list[spawn_index]
                vehicle = Vehicle(player, start, end, waypoints, colsensor, character_value, direction)
                self.character_dict[character_value] = True
                character_list.append(character_value)
                self.player_list.append(vehicle)

        next_state_list = []
        for index, vehicle in enumerate(self.player_list):
            tmp_state, _ = self.perception(vehicle)
            next_state_list.append(np.array(tmp_state))

        return next_state_list, character_list

    def step(self, actions):
        """
        :param actions: The action list for every agent
        :return:  next_state, reward, done for each agent
        """
        next_state_list = []
        next_done_list = []
        next_arrive_list = []
        next_character_list = []
        skip_num = 25
        slow_control = carla.VehicleControl()
        slow_control.throttle, slow_control.steer, slow_control.brake = 0.3, 0.0, 0.0
        control_list = []
        """
        for i in range(skip_num):
            for index, vehicle in enumerate(self.player_list):
                if not vehicle.is_arrive:
                    _, local_planner_map = self.perception(vehicle)
                    control_command, route_point, point, reward = self.controller.control(actions[index], vehicle,
                                                                                          vehicle.route,
                                                                                          local_planner_map)
                    vehicle.last_local_reward = reward
                    tmp_control = carla.ApplyVehicleControl(vehicle.player.id, control_command)
                    control_list.append(tmp_control)
                    vehicle.clock += 1
                    self.draw_waypoint(route_point, vehicle.character_value)
                else:
                    tmp_control = carla.ApplyVehicleControl(vehicle.player.id, slow_control)
                    control_list.append(tmp_control)
                    vehicle.clock += 1
            self.client.apply_batch(control_list)
            time.sleep(0.008)
        """ # old version code:
        for i in range(skip_num):
            for index, vehicle in enumerate(self.player_list):
                if not vehicle.is_arrive:
                    _, local_planner_map = self.perception(vehicle)
                    control_command, route_point, point, reward = self.controller.control(actions[index], vehicle,
                                                                                          vehicle.route, local_planner_map)
                    # self.draw_points(point, vehicle.character_value)
                    vehicle.last_local_reward = reward
                    vehicle.player.apply_control(control_command)
                    vehicle.clock += 1
                    self.draw_waypoint(route_point, vehicle.character_value)
                else:
                    vehicle.player.apply_control(slow_control)
                    vehicle.clock += 1
                time.sleep(0.001)
        # time.sleep(0.1)

        for index, vehicle in enumerate(self.player_list):
            tmp_state, local_planner_map = self.perception(vehicle)
            next_state_list.append(np.array(tmp_state))
            collision_done, reach_done, overtime_done = vehicle.check_done()
            next_done_list.append(collision_done + overtime_done)
            next_arrive_list.append(reach_done)
            next_character_list.append(vehicle.character_value)

        # print("next_arrive_list:", next_arrive_list)
        next_reward_list = self.cal_global_reward(next_done_list, next_arrive_list, self.last_arrive_list)
        print("next_reward:", next_reward_list)
        self.last_arrive_list = next_arrive_list

        destroy_list = []
        for i in range(len(next_done_list)):
            if next_done_list[i] == 1:
                destroy_list.append(self.player_list[i])

        self.destroy(destroy_list)
        for player in destroy_list:
            self.player_list.remove(player)
            del player

        # print(next_done_list)
        if sum(next_done_list) >= 1 or sum(next_arrive_list) == 8:
            next_done_list.append(1)
        else:
            next_done_list.append(0)
        return next_state_list, next_reward_list, next_done_list, next_character_list

    def perception(self, vehicle):
        occ_map = np.zeros((24, 24))
        vel_map = np.zeros((24, 24))

        now_transform = vehicle.player.get_transform()
        now_x, now_y, now_theta = now_transform.location.x, now_transform.location.y, now_transform.rotation.yaw
        vx = vehicle.player.get_velocity().x
        vy = vehicle.player.get_velocity().y
        now_speed = math.sqrt(vx ** 2 + vy ** 2)
        if vehicle.is_arrive:
            other_state = np.array([0./120.0, 1, 0, 0])
        else:
            if vehicle.direction == 0:
                other_state = np.array([vehicle.dist / 120.0, 1, 0, 0])
            elif vehicle.direction == 1:
                other_state = np.array([vehicle.dist / 120.0, 0, 1, 0])
            else:
                other_state = np.array([vehicle.dist / 120.0, 0, 0, 1])

        local_planner_map = []

        for other_vehicle in self.player_list:
            if other_vehicle != vehicle:
                other_location = other_vehicle.player.get_location()
                vx = other_vehicle.player.get_velocity().x
                vy = other_vehicle.player.get_velocity().y
                other_speed = math.sqrt(vx ** 2 + vy ** 2)

                rol = math.sqrt((other_location.x - now_x) ** 2 + (other_location.y - now_y) ** 2)
                theta = math.atan2(now_y - other_location.y, now_x - other_location.x)
                reltheta = theta - now_theta / 180 * math.pi
                rel_x = int((-rol * math.sin(reltheta) / 2.5) + np.sign((-rol * math.sin(reltheta)) / 2.5) * 0.5)
                rel_y = int((-rol * math.cos(reltheta) / 2.5) + np.sign((-rol * math.cos(reltheta)) / 2.5) * 0.5)
                local_planner_map.append([(-rol * math.sin(reltheta)), (-rol * math.cos(reltheta))])
                if -12 <= rel_x < 12 and -12 <= rel_y < 12:
                    occ_map[11 - rel_y][12 + rel_x] = 1.0
                    vel_map[11 - rel_y][12 + rel_x] = float(
                        min(1., max(0., other_speed / (2 * float(now_speed + 0.001)))))

        return np.concatenate((np.concatenate((np.reshape(occ_map, -1), np.reshape(vel_map, -1)), -1), other_state),
                              -1), local_planner_map

    def cal_reward(self):
        done_list = []
        reward_list = []
        for vehicle in self.player_list:
            a, b, c = vehicle.check_done()
            done_list.append([a, b, c])

        for index, vehicle in enumerate(self.player_list):
            reward = vehicle.last_local_reward + done_list[index][0] * (-10.0) + done_list[index][1] * 10.0 + \
                    done_list[index][2] * (-5.0)
            # reward = done_list[index][0] * (-10.0) + done_list[index][1] * 10.0 + done_list[index][2] * (-1.0)
            others_collision, others_reach = 0, 0
            for j, done in enumerate(done_list):
                if j != index:
                    others_collision += done[0]
                    others_reach += done[1]

            # reward -= 0.1 * vehicle.character_value * others_collision
            # reward += 0.1 * vehicle.character_value * others_reach

            reward_list.append(reward)

        return reward_list

    @staticmethod
    def cal_global_reward(is_collision, is_all_arrive, last_is_arrive):
        if sum(is_all_arrive)==8:
            return [10]
        elif sum(is_collision)>=1:
            return [-10]
        else:
            return [-0.01 + 2 * (sum(is_all_arrive) - sum(last_is_arrive))]

    @staticmethod
    def get_end_point(start, direction):
        end = carla.Transform().location
        if direction == 0:
            if start.x > 10:
                end.x = -1.6
                end.y = 40
            elif start.x < -10:
                end.x = 1.7
                end.y = -40.0
            elif start.y > 10:
                end.x = -40
                end.y = -1.7
            elif start.y < -10:
                end.x = 40
                end.y = 2.1
        elif direction == 1:
            if start.x > 10:
                end.x = -40.0
                end.y = -2.1
            elif start.x < -10:
                end.x = 40.0
                end.y = 2.1
            elif start.y > 10:
                end.x = 1.7
                end.y = -40
            elif start.y < -10:
                end.x = -1.7
                end.y = 40
        else:
            if start.x > 10:
                end.x = 1.6
                end.y = -40
            elif start.x < -10:
                end.x = -1.7
                end.y = 40.0
            elif start.y > 10:
                end.x = 40
                end.y = 2.1
            elif start.y < -10:
                end.x = -40
                end.y = -2.1

        return end

    @staticmethod
    def destroy(destroy_list):
        """
        :return: Destroy all agents
        """
        for x in destroy_list:
            x.collision_sensor.destroy()
            x.player.destroy()

    @staticmethod
    def gen_args():
        """
        :return: A module to generate default args
        """
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '-n', '--number-of-vehicles',
            metavar='N',
            default=10,
            type=int,
            help='number of vehicles (default: 10)')
        argparser.add_argument(
            '--rolename',
            metavar='NAME',
            default='hero',
            help='actor role name (default: "hero")')
        argparser.add_argument(
            '--filterv',
            metavar='PATTERN',
            default='vehicle.tesla.*',
            help='vehicles filter (default: "vehicle.tesla.*")')

        return argparser

    @staticmethod
    def get_random_spawn_points(num):
        spawn_list = []

        spawn_x = [1.70, -40, -1.7, 40, 1.70, -80, -1.7, 80]
        spawn_y = [40, 1.7, -40, -1.7, 80, 1.7, -80, -1.7]
        spawn_yaw = [-90, -1, 89, 179, -90, -1, 89, 179]
        rand_list = [0, 1, 2, 3]
        random.shuffle(rand_list)
        for i in range(num):
            rand_shift = random.random() * 40.0 - 20.0
            spawn_point = carla.Transform()
            spawn_point.location.x = spawn_x[i]
            spawn_point.location.y = spawn_y[i]
            spawn_point.location.z = 0.8
            spawn_point.rotation.yaw = spawn_yaw[i]

            if abs(spawn_point.location.x) > 10:
                spawn_point.location.x += rand_shift
            else:
                spawn_point.location.y += rand_shift
            spawn_list.append(spawn_point)

        return spawn_list

    def draw_waypoint(self, waypoint, character_value):
        if character_value == 0.0:
            color = carla.Color(255, 0, 0)  # 'red'
        elif character_value == 0.25:
            color = carla.Color(125, 75, 0)  # 'orange'
        elif character_value == 0.50:
            color = carla.Color(75, 125, 0)  # 'yellow'
        else:
            color = carla.Color(0, 255, 0)  # 'green'

        if waypoint:
            t = waypoint.transform
            begin = t.location + carla.Location(z=2.5)
            angle = math.radians(t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            self.world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=0.5, color=color)

    def draw_points(self, points, character_value):
        if character_value == 0.0:
            color = carla.Color(255, 0, 0)  # 'red'
        elif character_value == 0.25:
            color = carla.Color(125, 75, 0)  # 'orange'
        elif character_value == 0.50:
            color = carla.Color(75, 125, 0)  # 'yellow'
        else:
            color = carla.Color(0, 255, 0)  # 'green'
        for point in points:
            self.world.debug.draw_point(location=point.location, size=0.1, color=color, life_time=0.1)
