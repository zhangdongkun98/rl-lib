
import numpy as np
import torch


def func(self, rllib_data_func_name, *args, **kwargs):
    '''
        for torch.Tensor or np.ndarray
    '''

    new_dict = dict()
    for (key, value) in self.__dict__.items():
        if isinstance(value, (torch.Tensor, np.ndarray, Data)):
            _func = getattr(value, rllib_data_func_name)
            new_dict[key] = _func(*args, **kwargs)
        else:
            raise NotImplementedError
    return type(self)(**new_dict)


def attr(self, rllib_data_attr_name):
    '''
        for torch.Tensor or np.ndarray
    '''

    new_dict = dict()
    for (key, value) in self.__dict__.items():
        if isinstance(value, (torch.Tensor, np.ndarray, Data)):
            _attr = getattr(value, rllib_data_attr_name)
            new_dict[key] = _attr
        else:
            raise NotImplementedError
    return type(self)(**new_dict)


'''
import functools

def automatic(func_names, attr_names):
    def init(cls):
        for func_name in func_names:
            # def _func(self, func_name=func_name, *args, **kwargs):
            #     return func(self, func_name, *args, **kwargs)
            # _func = lambda self, *args, **kwargs: func(self, func_name, *args, **kwargs)
            _func = functools.partial(func, rllib_data_func_name=func_name)
            # setattr(cls, func_name, _func(func_name))
            setattr(cls, func_name, _func)
        for attr_name in attr_names:
            _attr = property(lambda self: attr(self, attr_name))
            setattr(cls, attr_name, _attr)
        return cls
    return init
'''


# @automatic(_func_names, _attr_names)
class Data(object):
    _func_numpy = []
    _func_torch = ['squeeze', 'unsqueeze', 'to', 'numpy']
    _func_names = [] + _func_numpy + _func_torch

    _attr_numpy = []
    _attr_torch = ['device']
    _attr_names = ['shape'] + _attr_numpy + _attr_torch

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        return

    def __str__(self):
        res = ''
        for (key, value) in self.__dict__.items():
            res += key + '=' + str(value) + ', '
        return self.__class__.__name__ + '({})'.format(res[:-2])

    def __repr__(self):
        return str(self)


    def to_dict(self):
        return self.__dict__

    def keys(self):
        return list(self.__dict__.keys())

    
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


    def __getattribute__(self, attribute):
        if attribute in Data._func_names:
            def make_interceptor():
                def _func(*args, **kwargs):
                    return func(self, attribute, *args, **kwargs)
                return _func
            return make_interceptor()
        elif attribute in Data._attr_names:
            _attr = lambda: attr(self, attribute)
            return _attr()
        else:
            return object.__getattribute__(self, attribute)

