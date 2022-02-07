
import os, glob
from os.path import join

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config, model_id=0):
        super(Model, self).__init__()
        self.model_id = model_id
        self.method_name = config.method_name
        self.model_dir = config.model_dir
        self.model_num = int(config.model_num)
        self.device = config.device
        self.dtype = torch.float32

        self.dim_state = config.get('dim_state', None)
        self.dim_action = config.get('dim_action', None)

    def load_model(self, model_num=None, model_dir=None):
        if model_dir == None:
            model_dir = self.model_dir
        if model_num == None:
            model_num = self.model_num

        model_dir = os.path.expanduser(model_dir)
        models_name = '_'.join([self.method_name.upper(), self.__class__.__name__, '*.pth'])
        file_paths = glob.glob(join(model_dir, models_name))
        file_names = [os.path.split(i)[-1] for i in file_paths]
        nums = [int(i.split('_')[-2]) for i in file_names]
        if model_num == -1:
            model_num = max(nums)

        print()
        print('[rllib.template.Model.load_model] model_dir: ', model_dir)
        print('[rllib.template.Model.load_model] models_name: ', models_name)
        print('[rllib.template.Model.load_model] file_paths length: ', len(file_paths))

        assert model_num in nums
        model_name = '_'.join([self.method_name.upper(), self.__class__.__name__, str(self.model_id), str(model_num), '.pth'])
        model_path = join(model_dir, model_name)
        print('[rllib.template.Model.load_model] load model: ', model_path)
        self.load_state_dict(torch.load(model_path))
    
    def save_model(self, path, iter_num):
        model_name = '_'.join([self.method_name.upper(), self.__class__.__name__, str(self.model_id), str(iter_num), '.pth'])
        model_path = join(path, model_name)
        torch.save(self.state_dict(), model_path)




# =============================================================================
# -- feature extraction -------------------------------------------------------
# =============================================================================


class FeatureExtractor(object):
    def __init__(self, config, model_id):
        self.dim_feature = config.dim_state
    def __call__(self, x, **kwargs):
        return x



# =============================================================================
# -- feature mapping ----------------------------------------------------------
# =============================================================================


class FeatureMapper(Model):
    def __init__(self, config, model_id, dim_input, dim_output):
        super().__init__(config, model_id)

        self.fm = nn.Sequential(
            nn.Linear(dim_input, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, dim_output),
        )
    
    def forward(self, x):
        return self.fm(x)

