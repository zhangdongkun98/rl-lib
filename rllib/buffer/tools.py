
from typing import List
from ..basic import Data


def stack_data(datas: List[Data]):
    data_keys = datas[0].keys()
    result = {}
    for key, i in zip(data_keys, zip(*datas)):
        if isinstance(i[0], Data):
            result[key] = stack_data(i)
        else:
            result[key] = i
    result = Data(**result)
    return result
