
import os, glob
from os.path import join

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config, model_id):
        super(Model, self).__init__()
        self.model_id = model_id
        self.method_name = config.method
        self.model_dir = config.model_dir
        self.model_num = int(config.model_num)
        self.device = config.device
        self.dtype = torch.float32

        self.dim_state = config.dim_state
        self.dim_action = config.dim_action

    def load_model(self, model_id=None):
        models_name = '_'.join([self.method_name.upper(), self.__class__.__name__, '*.pth'])
        file_paths = glob.glob(join(self.model_dir, models_name))
        file_names = [os.path.split(i)[-1] for i in file_paths]
        nums = [int(i.split('_')[-2]) for i in file_names]
        model_num = max(nums) if self.model_num == -1 else self.model_num
        assert model_num in nums
        if model_id == None: model_id = self.model_id
        model_name = '_'.join([self.method_name.upper(), self.__class__.__name__, str(model_id), str(model_num), '.pth'])
        model_path = join(self.model_dir, model_name)
        print('[load_model] load model: ', model_path)
        self.load_state_dict(torch.load(model_path))
    def save_model(self, path, iter_num):
        model_name = '_'.join([self.method_name.upper(), self.__class__.__name__, str(self.model_id), str(iter_num), '.pth'])
        model_path = join(path, model_name)
        torch.save(self.state_dict(), model_path)

    def __reduce_ex__(self, proto):
        return str(self)
