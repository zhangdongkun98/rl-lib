import carla_utils as cu

parse_yaml_file_unsafe = cu.parse_yaml_file_unsafe
create_dir = cu.basic.create_dir

setup_seed = cu.basic.setup_seed





def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_( (1 - t) * target_param.data + t * source_param.data )

def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)



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


