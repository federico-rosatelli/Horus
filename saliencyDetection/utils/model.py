from pathlib import Path
from . import *
import torch
import os
import cv2
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

class Items:
    def __init__(self,item:list[list[float]]) -> None:
        self.item = item
    
    def __call__(self) -> list[list[float]]:
        return self.item
    
    def __len__(self) -> int:
        return len(self.item)
    
    def __type__(self) -> type:
        return type(self.item)
    
    def size(self,idx=-1) -> list[int]|int:
        return [len(i) for i in self.item] if idx < 0 or idx > len(self.item) else len(self.item[idx])
    
    def avg(self) -> list[float]:
        return [sum(i)/len(i) for i in self.item]
    
    def max(self) -> list[float]:
        return [max(i) for i in self.item]
    
    def min(self) -> list[float]:
        return [min(i) for i in self.item]
    

def from_cv2_to_tensor(frame, size:None|tuple[int,int]=None):
    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pillow_array = Image.fromarray(bw_img)
    tensor = pil_to_tensor(pillow_array)/255
    if size:
        resize = Resize(size)
        tensor = resize(tensor)

    return tensor


class CheckPoint:
    """
    :class:`CheckPoint` class for implementing model saving and loading.
    """
    saved:any
    epoch:int
    state_dict:any
    optimizer:any
    loss:Items
    accuracy:Items

    def __init__(self,file_name:str,model_dir:str|None=None) -> None:
        
        root_dir = DIR.split(".")[0]
        model_dir = os.path.join(root_dir,MODELS_DIR) if not model_dir else model_dir
        
        self.file_name:str = os.path.join(model_dir,file_name)
    

    def _toDict(self,epoch,model,optimizer,loss,accuracy) -> dict:
        return {
            'epoch':epoch,
            'state_dict':model,
            'optimizer':optimizer,
            'tot_loss':Items(loss),
            'accuracy':Items(accuracy)
        }
        
    def save(self,epoch,model,optimizer,loss,accuracy):
        torch.save(self._toDict(epoch,model.state_dict(),optimizer,loss,accuracy),self.file_name)
        return self.load()

    def load(self):
        self.saved = torch.load(self.file_name)
        self.epoch = self.saved["epoch"]
        self.state_dict = self.saved["state_dict"]
        self.optimizer = self.saved["optimizer"]
        self.loss = self.saved["tot_loss"]
        self.accuracy = self.saved["accuracy"]

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
    
    def getLoss(self) -> Items:
        return self.loss
    
    def getAccuracy(self) -> Items:
        return self.accuracy
    
    def print(self) -> dict:
        return self._toDict(type(self.epoch),type(self.state_dict),type(self.optimizer),type(self.loss))
    
    def exists(self) -> bool:
        return Path(self.file_name).exists()
    

def accuracyPrediction(pred,lab) -> float:
    resize = Resize((lab.size(2),lab.size(3)))
    pred = resize(pred)
    _, predicted = torch.max(pred.data, 1)
    total = lab.size(0)
    correct = (predicted == lab).sum().item()
    return (correct/(lab.size(2)*lab.size(3)))/total