
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms



class NormalizeInverse(transforms.Normalize):
    '''
    Undoes the normalization and returns the reconstructed images in the input domain.
    '''

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())



class ToNumpyImage(object):
    def __init__(self):
        pass
    
    def __call__(self, x):
        assert len(x.shape) == 3

        x = x.permute(1,2,0)
        x = (x *255).numpy()
        x = x.astype(np.uint8)
        return x



def image_transforms(in_channels=3):
    _image_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*in_channels, (0.5,)*in_channels),
    ]
    return transforms.Compose(_image_transforms)


def image_transforms_reverse(in_channels=3):
    _image_transforms_reverse = [
        NormalizeInverse((0.5,)*in_channels, (0.5,)*in_channels),
        # transforms.ToPILImage(mode='RGB'),
        ToNumpyImage(),
    ]
    return transforms.Compose(_image_transforms_reverse)
