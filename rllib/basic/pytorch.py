
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import grad


def gradient(y, x, grad_outputs=None, allow_unused=False):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs == None:
        grad_outputs = torch.ones_like(y).to(y.device)
    _grad = grad(y, [x], grad_outputs=grad_outputs, create_graph=True, allow_unused=allow_unused)[0]
    return _grad


def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]
    
    x, y: torch.Size([batch_size, dim])
    """
    jac = torch.zeros(y.shape[1], x.shape[1]) 
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y).to(y.device)
        grad_outputs[:,i] = 1
        jac[i] = gradient(y, x, grad_outputs=grad_outputs)
    return jac




def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations

    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py

    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
    return





from torch.optim import Adam


class ModelArgMin(nn.Module):
    lr = 1e-1
    num_iter = 100

    def __init__(self, model: nn.Module, reverse=False):
        super().__init__()
        self.model = model
        self.sign = -1 if reverse else 1


    def forward(self, x0: torch.Tensor, args: torch.Tensor):
        x = x0.clone()
        x.requires_grad = True
        optimizer = Adam(params=[x], lr=self.lr)

        for i in range(self.num_iter):
            optimizer.zero_grad()

            value: torch.Tensor = self.forward_model(x, args) * self.sign
            loss = value.sum(dim=0)
            # loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

        return x.detach().clone()


    def forward_model(self, x0, args):
        return self.model(x0, args)




