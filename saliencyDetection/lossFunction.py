import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

#TODO

class MyMSELoss(Function):
    @staticmethod
    def forward(ctx, y_pred, y):    
        ctx.save_for_backward(y_pred, y)
        return ( (y - y_pred)**2 ).mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        grad_input = 2 * (y_pred - y) / y_pred.shape[0]        
        return grad_input, None