
import torch


def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs == None:
        grad_outputs = torch.ones_like(y).to(y.device)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


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
