
print('This is rllib.')

import numpy as np
import torch
np.set_printoptions(precision=6, linewidth=65536, suppress=True, threshold=np.inf)
torch.set_printoptions(precision=6, threshold=1000, edgeitems=None, linewidth=65536, profile=None, sci_mode=False)


from . import basic
from . import utils

from . import template
from . import buffer


'''methods'''
from . import a1c
from . import ppo_simple, ppo
from . import ppg

from . import dqn
from . import ddpg
from . import td3

from . import sac



from .evaluate import EvaluateSingleAgent
