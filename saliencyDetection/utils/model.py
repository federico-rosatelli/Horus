from pathlib import Path
from . import *
import torch
import os


class CheckPoint:
    saved:any
    epoch:int
    state_dict:any
    optimizer:any
    loss:list[int]

    def __init__(self,file_name:str,model_dir:str|None=None) -> None:
        
        root_dir = DIR.split(".")[0]
        model_dir = os.path.join(root_dir,MODELS_DIR) if not model_dir else model_dir
        
        self.file_name:str = os.path.join(model_dir,file_name)
    

    def _toDict(self,epoch,model,optimizer,loss) -> dict:
        return {
            'epoch':epoch,
            'state_dict':model,
            'optimizer':optimizer,
            'tot_loss':loss
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
    
    def exportConf(self,**args):
        expDict = {
            'file':self.file_name,
            'epoch':self.epoch,
            'state_dict':self.state_dict,
            'optimizer': self.optimizer,
            'tot_loss': self.loss
        }
        for key,val in args:
            expDict[key] = val
        return expDict
    
    def getEpoch(self) -> int:
        return self.epoch
    
    def getStateDict(self) -> any:
        return self.state_dict
    
    def getOptimizer(self) -> any:
        return self.optimizer
    
    def getLoss(self) -> list[int]:
        return self.loss
    
    def print(self) -> dict:
        return self._toDict(type(self.epoch),type(self.state_dict),type(self.optimizer),type(self.loss))
    
    def exists(self) -> bool:
        return Path(self.file_name).exists()
    