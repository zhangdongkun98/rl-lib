import carla
import math
import copy
import numpy as np

class LocalPlanner(object):
    def __init__(self):
        self.simsteps = 5
        self.sample_num = 5
        self.sim_delt = 0.25
        self.max_speed = 8

    def plan(self, set_v, set_w, now_transform, observe, waypoint):
        self.set_v = set_v
        self.set_w = set_w
        self.init_trans = now_transform

        vw_list = self.sample_vw(set_v, set_w)
        cost_list = []
        all_traj_list = []
        #print("set_v, set_w:", set_v, set_w)
        for (sample_v, sample_w) in vw_list:
            apply_v, apply_w = sample_v, sample_w
            traj_list = []
            now_transform = self.init_trans
            for i in range(self.simsteps):
                next_transform = self.step(apply_v, apply_w, now_transform)
                apply_v, apply_w = self.global_plan(next_transform, waypoint, apply_v)
                traj_list.append(next_transform)
                now_transform = next_transform

            all_traj_list.append(traj_list)
            sample_cost = self.cal_cost(traj_list, observe, waypoint, sample_v, sample_w)
            cost_list.append(sample_cost)

        optimal_v, optimal_w, reward = self.get_optimal_vw(vw_list, cost_list)
        #_, optimal_w = self.global_plan(self.init_trans, waypoint, optimal_v)
        points = self.get_optimal_points(cost_list, all_traj_list)

        #points = []
        #reward = 0
        #optimal_v, optimal_w = set_v, set_w
        return optimal_v, optimal_w, points, reward

    def sample_vw(self, set_v, set_w):
        v_list = [7.0/5.0 * set_v, 6.0/5.0 * set_v, set_v, 3.0/5.0 * set_v, 1.0/5.0 * set_v]
        #v_list = [set_v + 4, set_v + 2, set_v, set_v - 2, 0]
        w_list = [set_w]
        sample_vw_list = []
        for v in v_list:
            for w in w_list:
                sample_vw_list.append((v,w))
        return sample_vw_list

    def step(self, v, w, transform):
        next_transform = carla.Transform()
        x, y, yaw = self.extract_transform(transform)
        next_transform.location.x = x + v * self.sim_delt * math.cos(math.radians(yaw))
        next_transform.location.y = y + v * self.sim_delt * math.sin(math.radians(yaw))
        next_transform.rotation.yaw = w * self.sim_delt + yaw

        return next_transform

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
        k_theta = 0.5
        k_e = 0.1

        v_ref = apply_v

        apply_w = v_ref * k_s * math.cos(theta_error) / (1 - k_s * dist_error) \
                  - (k_theta * abs(v_ref)) * theta_error + \
                  (k_e * v_ref * math.sin(theta_error) / (theta_error + 0.0001)) * dist_error
        apply_w = apply_w * 180 / math.pi
        return apply_v, apply_w

    def cal_cost(self, traj_list, observe, waypoint, sample_v, sample_w):
        obs_cost = self.cal_obs_cost(traj_list, observe)
        smooth_cost = (abs(sample_v - self.set_v)*10 + abs(sample_w - self.set_w))
        speed_cost = -self.set_v * 5
        direct_cost = self.cal_direct_cost(traj_list[-1], waypoint)
        final_cost = 0.2 * obs_cost + 0.05 * direct_cost + 0.5 * smooth_cost + 0.25 * speed_cost
        #final_cost = smooth_cost

        return final_cost

    def get_optimal_vw(self, vw_list, cost_list):
        index = cost_list.index(min(cost_list))

        reshape_cost = min(cost_list)/20
        reward = -((np.exp(reshape_cost) - np.exp(-reshape_cost)) / (np.exp(reshape_cost) + np.exp(-reshape_cost)))
        (optimal_v, optimal_w) = vw_list[index]
        #print("v_index:", index/5, "w_index", index%5)

        if optimal_v > self.max_speed:
            optimal_v = self.max_speed

        #print(cost_list)
        #print(vw_list)
        #print("optimal", optimal_v, optimal_w)
        return optimal_v, optimal_w, reward

    def cal_obs_cost(self, traj_list, observe):
        weight = 1.0
        obs_cost = 0
        for trans in traj_list:
            nearest_dist = self.find_nearest_obs(trans, observe)
            weight *= 0.5
            if nearest_dist <= 4.0:
                nearest_dist = -1000 * 1.0 / (nearest_dist + 1)
            obs_cost += weight * (-1.0) * nearest_dist

        return obs_cost

    def cal_smooth_cost(self, traj):
        return 0

    def cal_direct_cost(self, traj, waypoint):

        #_, _, now_yaw = self.extract_transform(traj)
        #_, _, waypoint_yaw = self.extract_transform(waypoint.transform)
        #diff = abs(now_yaw - waypoint_yaw)
        diff = 0
        return diff

    def extract_transform(self, transform):
        return transform.location.x, transform.location.y, transform.rotation.yaw

    def get_dist_error(self, location, waypoint):
        """
        :param location:
        :param waypoint:
        :return:dist_error
        """
        rol = math.sqrt((location.x - waypoint.transform.location.x)**2+(location.y-waypoint.transform.location.y)**2)
        theta = math.atan2((waypoint.transform.location.y-location.y),(waypoint.transform.location.x-location.x))
        reltheta = theta - (waypoint.transform.rotation.yaw)/180 * math.pi
        dist_error = rol * math.sin(reltheta)

        return dist_error

    def find_nearest_obs(self, trans, observe):
        now_x, now_y, now_yaw = self.extract_transform(trans)
        init_x, init_y, init_yaw = self.extract_transform(self.init_trans)
        #print(now_x, now_y, now_yaw, init_x, init_y, init_yaw)

        rol = math.sqrt((now_x - init_x) ** 2 + (now_y - init_y) ** 2)
        # theta = math.atan2(othercar.y-now_location.y, -(othercar.x-now_location.x))
        theta = math.atan2(init_y - now_y, init_x - now_x)
        reltheta = theta - math.radians(init_yaw)

        #print("theta:", theta, "reltheta:", reltheta)
        relX = - rol * math.sin(reltheta)
        relY = - rol * math.cos(reltheta)

        if relX >= 0:
            indexX = int(relX + 0.5)
        else:
            indexX = int(relX - 0.5)
        if relY >= 0:
            indexY = int(relY + 0.5)
        else:
            indexY = int(relY - 0.5)

        #print("index:", indexX, indexY)
        min_dist = 100
        for vehicle in observe:
            dist = math.sqrt((indexY - vehicle[1])**2 + 2 * (indexX - vehicle[0])**2)
            if dist < min_dist:
                min_dist = dist

        #print("indexX, indexY:", indexX, indexY, "min_dist", min_dist, "minij:", min_i, min_j)
        return min_dist

    def get_optimal_points(self, cost_list, traj_list):
        return traj_list[cost_list.index(min(cost_list))]
        #return traj_list[10]


