
import torch


class Data(object):
    '''
        carla_utils has this class too.
    '''
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        return

    def __str__(self):
        res = self.__class__.__name__ + '('
        for (key, value) in self.__dict__.items():
            res += key + '=' + str(value) + ', '
        return res[:-2] + ')'

    def to(self, device):
        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                new_dict[key] = value.to(device)
            else:
                raise NotImplementedError
        return type(self)(**new_dict)


class Experience(Data):
    pass

