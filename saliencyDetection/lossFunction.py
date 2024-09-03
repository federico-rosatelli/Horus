import torch
import torch.nn as nn
import numpy as np
import logging
logging.getLogger("numpy").propagate = False

U = 0.5

class HorusLossFunction(nn.Module):
    """
    Custom Loss Function implementing Soft & Hard Loss
    """
    @staticmethod
    def forward(s_res, t_res, g_lab):    
        return Loss(s_res, t_res, g_lab)


def Loss(student_predict,teacher_predict,ground_teacher) -> float:
    #teacher_s_result,student_s_result,ground_s_map = spatial_result
    teacher_s_resize = np.resize(teacher_predict.detach().numpy(),(256,256))
    ground_s_resize = np.resize(ground_teacher.detach().numpy(),(256,256))
    
    slS = SHLoss(student_predict,torch.from_numpy(teacher_s_resize))               # soft loss (student_pred_spt, teacher_pred_spt)
    hlS = SHLoss(student_predict,torch.from_numpy(ground_s_resize))                # hard loss (student_pred_stp, teacher_ground_spt)
    # [1,256,1,256]
    L1 =  U*slS + (1-U)*hlS

    return L1


class HorusSpatioTemporalLoss(nn.Module):
    """
    Custom Loss Function implementing SpatioTemporal Hard Loss
    """

    @staticmethod
    def forward(sp_res, g_lab):
        return SHLoss(sp_res,g_lab)


def SHLoss(student_result:torch.Tensor,teacher_ground_result:torch.Tensor) -> float:
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

