from pathlib import Path
from . import *
import torch
import os
import cv2
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

class LossItems:
    def __init__(self,loss:list[list[float]]) -> None:
        self.loss = loss
    
    def __call__(self) -> list[list[float]]:
        return self.loss
    
    def __len__(self) -> int:
        return len(self.loss)
    
    def __type__(self) -> type:
        return type(self.loss)
    
    def size(self,idx=-1) -> list[int]|int:
        return [len(i) for i in self.loss] if idx < 0 or idx > len(self.loss) else len(self.loss[idx])
    
    def avg(self) -> list[float]:
        return [sum(i)/len(i) for i in self.loss]
    
    def max(self) -> list[float]:
        return [max(i) for i in self.loss]
    
    def min(self) -> list[float]:
        return [min(i) for i in self.loss]
    
def from_cv2_to_tensor(frame, size:None|tuple[int,int]=None):
    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pillow_array = Image.fromarray(bw_img)
    tensor = pil_to_tensor(pillow_array)/255
    if size:
        resize = Resize(size)
        tensor = resize(tensor)

    return tensor

class CheckPoint:
    saved:any
    epoch:int
    state_dict:any
    optimizer:any
    loss:LossItems

    def __init__(self,file_name:str,model_dir:str|None=None) -> None:
        
        root_dir = DIR.split(".")[0]
        model_dir = os.path.join(root_dir,MODELS_DIR) if not model_dir else model_dir
        
        self.file_name:str = os.path.join(model_dir,file_name)
    

    def _toDict(self,epoch,model,optimizer,loss) -> dict:
        return {
            'epoch':epoch,
            'state_dict':model,
            'optimizer':optimizer,
            'tot_loss':LossItems(loss)
        }
        
    def save(self,epoch,model,optimizer,loss) -> None:
        torch.save(self._toDict(epoch,model.state_dict(),optimizer,loss),self.file_name)
        return

    def load(self):
        self.saved = torch.load(self.file_name)
        self.epoch = self.saved["epoch"]
        self.state_dict = self.saved["state_dict"]
        self.optimizer = self.saved["optimizer"]
        self.loss = self.saved["tot_loss"]

        return self
    
    def exportConf(self,**kwargs):
        expDict = {
            'file':self.file_name,
            'epoch':self.getEpoch(),
            'state_dict':self.getStateDict(),
            'optimizer': self.getOptimizer(),
            'tot_loss': self.getLoss()
        }
        for key,val in kwargs.items():
            expDict[key] = val
        return expDict
    
    def getEpoch(self) -> int:
        return self.epoch
    
    def getStateDict(self) -> any:
        return self.state_dict
    
    def getOptimizer(self) -> any:
        return self.optimizer
    
    def getLoss(self) -> LossItems:
        return self.loss
    
    def print(self) -> dict:
        return self._toDict(type(self.epoch),type(self.state_dict),type(self.optimizer),type(self.loss))
    
    def exists(self) -> bool:
        return Path(self.file_name).exists()
    