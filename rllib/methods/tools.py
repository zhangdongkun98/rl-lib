

import torch.nn as nn



def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        try: nn.init.constant_(m.bias, 0.01)
        except: pass
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if name.startswith('weight'): nn.init.orthogonal_(param)
    return
