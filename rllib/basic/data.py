
import numpy as np
import copy
import torch


def func(self, rllib_data_func_name, *args, **kwargs):
    '''
        for torch.Tensor or np.ndarray
    '''

    new_dict = dict()
    for (key, value) in self.__dict__.items():
        # if isinstance(value, (torch.Tensor, np.ndarray, Data)):
        if hasattr(value, rllib_data_func_name):
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
        # if isinstance(value, (torch.Tensor, np.ndarray, Data)):
        if hasattr(value, rllib_data_attr_name):
            _attr = getattr(value, rllib_data_attr_name)
            new_dict[key] = _attr
        else:
            # raise NotImplementedError
            new_dict[key] = 'NotImplementedError'
    return type(self)(**new_dict)




class BaseData(object):

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        return
    
    def merge(self, data):
        self.update(**data.to_dict())

    def __add__(self, data):
        """
            warning: use copy.copy rather than copy.deepcopy
        """
        d = copy.copy(self)
        d.merge(data)
        return d

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
        """
            todo: recursion
        """
        return self.__dict__

    def keys(self):
        return list(self.__dict__.keys())
    def dvalues(self):
        return list(self.__dict__.values())


    def pop(self, key):
        return self.__dict__.pop(key)



class Data(BaseData):
    _func_numpy = []
    _func_torch = ['squeeze', 'unsqueeze', 'cpu', 'numpy', 'detach', 'requires_grad_', 'clone', 'max', 'expand', 'reshape']
    _func_names = ['repeat'] + _func_numpy + _func_torch

    _attr_numpy = []
    _attr_torch = ['device', 'requires_grad', 'dtype', 'values']
    _attr_names = ['shape'] + _attr_numpy + _attr_torch


    # =============================================================================
    # -- dict ---------------------------------------------------------------------
    # =============================================================================

    def to(self, *args, **kwargs):
        """
            for torch.Tensor
        """
        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.to(*args, **kwargs)
            elif isinstance(value, torch.Tensor):
                new_dict[key] = value.to(*args, **kwargs)
            elif isinstance(value, list):
                new_dict[key] = [v.to(*args, **kwargs) for v in value]
            else:
                raise NotImplementedError
                # new_dict[key] = 'NotImplementedError'
        return type(self)(**new_dict)


    def stack(self, *args, **kwargs):
        """
            for torch.Tensor
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.stack(*args, **kwargs)
            elif all([isinstance(v, torch.Tensor) for v in value]):
                new_dict[key] = torch.stack(value, *args, **kwargs)
            else:
                new_dict[key] = torch.as_tensor(value)
        return type(self)(**new_dict)

    def cat(self, *args, **kwargs):
        """
            for torch.Tensor
        """

        new_dict = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Data):
                new_dict[key] = value.cat(*args, **kwargs)
            # elif isinstance(value, torch.Tensor):
            elif all([isinstance(v, torch.Tensor) for v in value]):
                new_dict[key] = torch.cat(value, *args, **kwargs)
            else:
                new_dict[key] = torch.as_tensor(value)
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


    def __getitem__(self, key):
        if isinstance(key, slice):
            new_dict = dict()
            for (_key, _value) in self.__dict__.items():
                new_dict[_key] = _value[key]
            return type(self)(**new_dict)
        elif all([isinstance(k, slice) for k in key]):
            raise NotImplementedError("Comming soon!")

        elif isinstance(key, tuple):
            new_dict = dict()
            for (_key, _value) in self.__dict__.items():
                new_dict[_key] = _value[key]
            return type(self)(**new_dict)
        
        elif isinstance(key, int):
            print('[rllib.basic.Data::__getitem__] int key will be deprecated')
            return getattr(self, self.keys()[key])
        else:
            raise NotImplementedError
        return


    def memory_usage(self):
        import sys
        from pympler import asizeof  ### pip install pympler==1.0.1
        size = 0
        for value in self:
            if isinstance(value, Data):
                size += value.memory_usage()
            elif isinstance(value, torch.Tensor):
                size += sys.getsizeof(value.storage())
            else:
                # size += sys.getsizeof(value)
                size += asizeof.asizeof(value)
        return size

