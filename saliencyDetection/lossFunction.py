import torch
import torch.nn as nn
import torch.nn.functional as F


U = 0.5

class HorusLossFunction(nn.Module):
    """
    Custom Loss Function implementing Soft & Hard Loss
    """
    @staticmethod
    def forward(ys_pred, yt_pred, y):    
        #ctx.save_for_backward(ys_pred,yt_pred , y)
        return Loss(ys_pred,yt_pred,y)
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     y_pred, y = ctx.saved_tensors
    #     grad_input = 2 * (y_pred - y) / y_pred.shape[0]     #da modificare   
    #     return grad_input, None

def Loss(teacher_result:torch.Tensor,student_result:torch.Tensor,ground_map:torch.Tensor) -> float:
    # TODO: ??? reshape [1, 720, 1, 1280] to [1,256,3,256]
    print(teacher_result.size(),student_result.size())
    teacher_resize = teacher_result.view(1,256,3,1,1280)
    ground_resize = ground_resize.view(1,256,3,1,1280)
    teacher_resize = F.interpolate(teacher_result,size=(256, 256), mode='bilinear', align_corners=False)
    ground_resize = F.interpolate(ground_map,size=(256, 256), mode='bilinear', align_corners=False)
    sl = SHLoss(student_result,teacher_resize)
    hl = SHLoss(student_result,ground_resize)

    return U*sl + (1-U)*hl



def SHLoss(student_result:torch.Tensor,teacher_result:torch.Tensor) -> float:
    """
    Soft & Hard Loss
    >>> 1/(w*h) * norm((s-t),2)
    """
    w,h = 256,256
    diff = student_result - teacher_result
    norm = torch.norm(diff, p=2)
    return (
        1/(h * w)
    ) * norm.item()

