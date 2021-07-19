
import numpy as np

import torch


class Data(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        return
    

    def __str__(self):
        res = self.__class__.__name__ + '('
        for (key, value) in self.__dict__.items():
            res += key + '=' + str(value) + ', '
        return res[:-2] + ')'


    def to_dict(self):
        '''
            for Any
        '''
        
        return self.__dict__

    def to(self, device):
        '''
            for torch.Tensor
        '''

        new_dict = dict()
        for (key, value) in self.__dict__.items():            
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                new_dict[key] = value.to(device)
            else:
                raise NotImplementedError
        return type(self)(**new_dict)


    def squeeze(self, *args):
        '''
            for torch.Tensor
        '''

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                new_dict[key] = value.squeeze(*args)
            else:
                raise NotImplementedError
        return type(self)(**new_dict)

    def unsqueeze(self, *args):
        '''
            for torch.Tensor
        '''

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                new_dict[key] = value.unsqueeze(*args)
            else:
                raise NotImplementedError
        return type(self)(**new_dict)


    def numpy(self):
        '''
            for torch.Tensor
        '''

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, torch.Tensor) or isinstance(value, Data):
                new_dict[key] = value.numpy()
            else:
                raise NotImplementedError
        return type(self)(**new_dict)

    
    def to_tensor(self):
        '''
            for np.ndarray
        '''

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, np.ndarray):
                new_dict[key] = torch.from_numpy(value)
            elif isinstance(value, Data):
                new_dict[key] = value.to_tensor()
            else:
                raise NotImplementedError
        return type(self)(**new_dict)


    @property
    def shape(self):
        '''
            for torch.Tensor and np.ndarray
        '''

        res = self.__class__.__name__ + '('
        for (key, value) in self.__dict__.items():
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray) or isinstance(value, Data):
                res += key + '=' + str(value.shape) + ', '
            else:
                raise NotImplementedError
        return res[:-2] + ')'
