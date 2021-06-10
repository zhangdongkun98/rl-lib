
import numpy as np

import torch

from method_maddpg.model import Actor, Critic, CNN, FC

p_map =np.random.rand(24,24)
v_map =np.random.rand(24,24)

batch_size = 64

model = CNN(2,256)

data = torch.from_numpy(np.stack((p_map, v_map)))
data = data.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(torch.float32)

print('input: ', data.shape)

model(data)
