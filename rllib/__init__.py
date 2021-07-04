
print('This is rllib.')

import numpy as np
import torch
np.set_printoptions(precision=6, linewidth=65536, suppress=True)
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)


from . import basic

from . import args, gallery

from . import utils

from . import template

'''methods'''

from . import ppo

from . import td3
from . import ddpg
from . import dqn
