
import torch
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



def image_transforms(in_channels=3):
    _image_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*in_channels, (0.5,)*in_channels),
    ]
    return transforms.Compose(_image_transforms)


def image_transforms_reverse(in_channels=3):
    _image_transforms_reverse = [
        NormalizeInverse((0.5,)*in_channels, (0.5,)*in_channels),
        transforms.ToPILImage(mode='RGB')
    ]
    return transforms.Compose(_image_transforms_reverse)
