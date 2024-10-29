import torch
import torch.nn as nn
import logging
from torchvision.transforms import Resize
import inspect
logging.getLogger("numpy").propagate = False

U = 0.5



def getLossFunction(name:str) -> any:
    
    for cls_name,obj in inspect.getmembers(nn):
        if inspect.isclass(obj):
            if name.lower() == cls_name.lower():
                return obj
    
    return nn.MSELoss


class HorusLossFunction(nn.Module):
    """
    Custom Loss Function implementing Soft & Hard Loss
    """
    @staticmethod
    def forward(s_res, t_res, g_lab):    
        return Loss(s_res, t_res, g_lab)


def Loss(student_predict,teacher_predict,ground_teacher):
    resize = Resize((student_predict.size(2),student_predict.size(3)))
    teacher_resize = resize(teacher_predict)
    ground_resize = resize(ground_teacher)
    
    slS = SHLoss(student_predict,teacher_resize)               # soft loss (student_pred_spt, teacher_pred_spt)
    hlS = SHLoss(student_predict,ground_resize)                # hard loss (student_pred_stp, teacher_ground_spt)
    
    return (U*slS) + ((1-U)*hlS)


class HorusSpatioTemporalLoss(nn.Module):
    """
    Custom Loss Function implementing SpatioTemporal Hard Loss
    """

    @staticmethod
    def forward(sp_res, g_lab):
        return SHLoss(sp_res,g_lab)


def SHLoss(student_result:torch.Tensor,teacher_ground_result:torch.Tensor):
    """
    Soft & Hard Loss
    >>> 1/(w*h) * norm((s-t),2)
    """
    w,h = 256,256
    diff = student_result - teacher_ground_result
    norm = torch.norm(diff, p="fro")    # Frobenius norm produces the same result as `p=2` in all cases
                                        # except when `dim` is a list of three or more dims, in which
                                        # case Frobenius norm throws an error.

    return (
        1/(h * w)
    ) * norm    # !!!!.item()

