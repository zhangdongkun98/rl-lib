import carla_utils as cu
from carla_utils import carla

import numpy as np


class VehiclesVisualizerInNumpy(object):
    def __init__(self, resolution, g_range, l_range):
        """
        
        
        Args:
            resolution: m/pix
        
        Returns:
            
        """
        self.resolution, self.g_range, self.l_range = resolution, g_range, l_range
        self.g_height, self.g_width = int(g_range.y / resolution), int(g_range.x / resolution)
        self.l_height, self.l_width = int(l_range.y / resolution), int(l_range.x / resolution)
        self.min, self.l_max = np.array([0,0], dtype=np.int64), np.array([self.l_width, self.l_height], dtype=np.int64)
        self.l_valid = lambda p: (p >= self.min) & (p < self.l_max)
        self.x_array = np.arange(self.g_width).reshape(1,self.g_width).repeat(self.g_height, axis=0)
        self.y_array = np.arange(self.g_height).reshape(self.g_height,1).repeat(self.g_width, axis=1)
    
    
    def global_mask_indexes(self, vehicles):
        mask_indexes = dict()
        for vehicle in vehicles:
            boundary = self._draw_vehicle(vehicle)
            mask_index = self._mask_index(boundary)
            mask_indexes[vehicle.id] = mask_index
        return mask_indexes

    def crop(self, global_mask_indexes, vehicle):
        bbx = vehicle.bounding_box.extent
        expand = carla.Vector2D((self.l_range.y-self.resolution)/2-bbx.y, (self.l_range.x-self.resolution)/2-bbx.x)
        boundary = self._draw_vehicle(vehicle, expand)

        theta_vehicle = np.deg2rad(vehicle.get_transform().rotation.yaw)
        htm = cu.basic.HomogeneousMatrixInverse2D.xytheta([boundary[0][0], boundary[0][1], np.pi/2 - theta_vehicle])

        local_mask_indexes = dict()
        for key, global_index in global_mask_indexes.items():
            local_index = np.dot(htm, np.vstack((global_index.T[::-1,:], np.ones((1,global_index.shape[0])))))[:2].T
            local_index = np.round(local_index).astype(np.int64)
            local_mask = self.l_valid(local_index)
            local_index = local_index[local_mask[:,0] & local_mask[:,1]]
            local_mask_indexes[key] = local_index[:,::-1]
        
        return local_mask_indexes


    def _mask_index(self, boundary):
        """
        
        
        Args:
            boundary: counter-clockwise
        
        Returns:
            
        """

        num = boundary.shape[0]
        condition = True
        for i in range(num):
            p1, p2 = boundary[i], boundary[(i+1)%num]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            condition &= ((y2-y1)*self.x_array - (x2-x1)*self.y_array -(y2-y1)*x1 + (x2-x1)*y1) > 0
        mask_index = np.argwhere(condition)
        ### Note that the shape of mask_index is (N,2), where (N,0) represents y/height axis. 
        return mask_index
    
    def _draw_vehicle(self, vehicle, expand=carla.Vector2D(0,0)):
        vertices = cu.ActorVertices.d2(vehicle, expand)
        vertices[:,1] *= -1
        boundary = vertices / self.resolution + np.array([[self.g_width/2, self.g_height/2]])
        return boundary
