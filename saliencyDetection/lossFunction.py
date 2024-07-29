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
    def forward(s_res, t_res):    
        return Loss(s_res,t_res)


def Loss(spatial_result,temporal_result) -> float:
    teacher_s_result,student_s_result,ground_s_map = spatial_result
    teacher_s_resize = np.resize(teacher_s_result.detach().numpy(),(256,256))
    ground_s_resize = np.resize(ground_s_map.detach().numpy(),(256,256))
    
    slS = SHLoss(student_s_result,torch.from_numpy(teacher_s_resize))               # soft loss (student_pred_spt, teacher_pred_spt)
    hlS = SHLoss(student_s_result,torch.from_numpy(ground_s_resize))                # hard loss (student_pred_stp, teacher_ground_spt)

    L1 =  U*slS + (1-U)*hlS

    teacher_t_result,student_t_result,ground_t_map = temporal_result
    teacher_t_resize = np.resize(teacher_t_result.detach().numpy(),(256,256))
    ground_t_resize = np.resize(ground_t_map.detach().numpy(),(256,256))
    
    slT = SHLoss(student_t_result,torch.from_numpy(teacher_t_resize))               # soft loss (student_pred_tem, teacher_pred_tem)
    hlT = SHLoss(student_t_result,torch.from_numpy(ground_t_resize))                # hard loss (student_pred_tem, teacher_ground_tem)

    L2 =  U*slT + (1-U)*hlT

    return (
            (L1**2)+
            (L2**2)
        )/2
    # s = U*sls + (1-U)*hls
    # t = U*slt + (1-U)*hlt
    # return math.sqrt(s**2 + t**2)



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
    ) * norm    # !!!!.item()

