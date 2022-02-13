
import os
import numpy as np
import random
import copy

import torch
import torch.nn as nn


from .system import prefix


def setup_seed(seed):
    print(prefix(__name__) + 'seed is: ', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



'''
https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
'''
def pi2pi(theta):
    if isinstance(theta, np.ndarray) or isinstance(theta, float):
        return pi2pi_numpy(theta)
    elif isinstance(theta, torch.Tensor):
        return pi2pi_numpy(theta)
    else: raise NotImplementedError
    

def pi2pi_numpy(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def pi2pi_tensor(theta):
    """
    Normalize the `theta` to have a value in [-pi, pi]

    Args:
        theta: Tensor of angles of shape N
    """
    TWO_PI = 2 * np.pi
    theta = torch.fmod(torch.fmod(theta, TWO_PI) + TWO_PI, TWO_PI)
    return torch.where(theta > np.pi, theta - TWO_PI, theta)


def sincos2rad(sin, cos):
    if isinstance(cos, np.ndarray) or isinstance(cos, float):
        package = np
    elif isinstance(cos, torch.Tensor):
        package = torch
    else: raise NotImplementedError
    
    theta = package.arctan(sin / cos)

    w = package.sign(package.cos(theta) * cos) + package.sign(package.sin(theta) * sin) - 2
    theta = pi2pi(theta - w * np.pi / 4)
    return theta



def np_dot(*args):
    res = args[0]
    for arg in args[1:]:
        res = np.dot(res, arg)
    return res

def int2onehot(index, length):
    '''
    Args:
        index: (batch_size,)
    return: numpy.array (length,)
    '''
    if isinstance(index, torch.Tensor):
        return nn.functional.one_hot(index.to(torch.int64), length).to(index.dtype)
    elif isinstance(index, np.ndarray):
        return np.eye(length)[index.reshape(-1)]
    elif isinstance(index, int):
        return np.eye(1, length, k=index)[0]
    else: raise NotImplementedError
    
def onehot2int(one_hot):
    if isinstance(one_hot, torch.Tensor):  ## will squeeze one dimension
        return torch.argmax(one_hot, dim=-1)
    else: raise NotImplementedError; return np.argmax(one_hot)


def prob2onehot(prob, length):
    '''
    Args:
        prob: (batch_size, length)
    '''
    if isinstance(prob, torch.Tensor):
        return nn.functional.one_hot(torch.argmax(prob, dim=1), length).to(prob.dtype)
    else: raise NotImplementedError




def numpy_gather(ndarray, dim, index):
    """
        Deprecated because of the risk of memory leak.
        Use numpy.take_along_axis instead.
    """
    tensor = torch.from_numpy(ndarray)
    index_index = torch.from_numpy(index)
    res = torch.gather(tensor, dim=dim, index=index_index)
    return res.numpy()



def pad_tensor(data: torch.Tensor, pad_size: torch.Size, pad_value=np.inf):
    """
    Args:
        data, pad_size: torch.Size([batch_size, dim_elements, dim_points, dim_features])
    """
    res = torch.full(pad_size, pad_value, dtype=data.dtype, device=data.device)

    if len(pad_size) == 3:
        batch_size, dim_elements, dim_points = data.shape
        res[:batch_size, :dim_elements, :dim_points] = data
    elif len(pad_size) == 4:
        batch_size, dim_elements, dim_points, dim_features = data.shape
        res[:batch_size, :dim_elements, :dim_points, :dim_features] = data
    else:
        raise NotImplementedError
    return res


def pad_array(data: np.ndarray, pad_size: tuple, pad_value=np.inf):
    res = np.full(pad_size, pad_value, dtype=data.dtype)

    if len(pad_size) == 2:
        d1, d2 = data.shape
        res[:d1, :d2] = data
    elif len(pad_size) == 3:
        d1, d2, d3 = data.shape
        res[:d1, :d2, :d3] = data
    elif len(pad_size) == 4:
        d1, d2, d3, d4 = data.shape
        res[:d1, :d2, :d3, :d4] = data
    else:
        raise NotImplementedError
    return res






import matplotlib.pyplot as plt
def plot_arrow(x, y, theta, length=1.0, width=0.5, fc='r', ec='k'):  # pragma: no cover
    plt.arrow(x, y, length * np.cos(theta), length * np.sin(theta), fc=fc, ec=ec, head_width=width, head_length=width)




def fig2array(fig):
    """
        fig = plt.figure()
        image = fig2array(fig)
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)[:,:,:3]
    return image




def list_del(list_to_delete, delete_index_list):
    dil = copy.copy(delete_index_list)
    dil.sort()
    dil.reverse()
    for index in dil:
        del list_to_delete[index]

def flatten_list(input_list):
    """
    
    
    Args:
        input_list: 2-d list
    
    Returns:
        1-d list
    """
    
    output_list = []
    for i in input_list: output_list.extend(i)
    return output_list


def calculate_quadrant(point):
    """
    
    
    Args:
        point: contains attribute x, y
    
    Returns:
        int
    """

    if point.x > 0 and point.y > 0:
        quadrant = 1
    elif point.x < 0 and point.y > 0:
        quadrant = 2
    elif point.x < 0 and point.y < 0:
        quadrant = 3
    elif point.x > 0 and point.y < 0:
        quadrant = 4
    else:
        quadrant = 0
    return quadrant




def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n
