
import carla_utils as cu

import torch
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)

from method_maddpg.model import Actor

config = cu.parse_yaml_file_unsafe('./config/carla.yaml')

a = Actor(0, config)

a.load_model()

param = a.parameters()

for i, c in enumerate(param):
    c.data
    print(c)
    print('------------------------------')