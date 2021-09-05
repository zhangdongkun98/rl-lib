
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
            # raise NotImplementedError
            new_dict[key] = 'NotImplementedError'
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
            # raise NotImplementedError
            new_dict[key] = 'NotImplementedError'
    return type(self)(**new_dict)


class Data(object):
    _func_numpy = []
    _func_torch = ['squeeze', 'unsqueeze', 'to', 'numpy']
    _func_names = ['repeat'] + _func_numpy + _func_torch

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

    def __iter__(self):
        """
            warning: limited use
        """
        for (key, value) in self.__dict__.items():
            # yield {key: value}
            yield value


    # =============================================================================
    # -- dict ---------------------------------------------------------------------
    # =============================================================================

    def to_dict(self):
        return self.__dict__

    def keys(self):
        return list(self.__dict__.keys())

    def pop(self, key):
        return self.__dict__.pop(key)


    # =============================================================================
    # -- dict ---------------------------------------------------------------------
    # =============================================================================

    def stack(self, *args, **kwargs):
        """
            for torch.Tensor
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.stack(*args, **kwargs)
            else:
                new_dict[key] = torch.stack(value, *args, **kwargs)
        return type(self)(**new_dict)        

    def cat(self, *args, **kwargs):
        """
            for torch.Tensor
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.cat(*args, **kwargs)
            else:
                new_dict[key] = torch.cat(value, *args, **kwargs)
        return type(self)(**new_dict)     


    
    def to_tensor(self):
        """
            for np.ndarray
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, np.ndarray):
                new_dict[key] = torch.from_numpy(value)
            elif isinstance(value, Data):
                new_dict[key] = value.to_tensor()
            else:
                # raise NotImplementedError
                new_dict[key] = torch.tensor(value)
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

