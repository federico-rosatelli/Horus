import logging
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False
from . import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from saliencyDetection.utils.model import CheckPoint


class Horus:
    """
    Horus Class Prediction.

    Args:
        model_class: any Model Class.
        model_file: string, name of the model file saved (eg. `pt` files).
        state_dict: boolean, if the model file is a state dictionary.
        device: string, `cpu`|`cuda` from the config file
    """
    HOME_PATH:str = f"{DIR}/{MODEL_DIR}/"
    
    def __init__(self,model_class:any,model_file:str="horus_model.pt",device:str="cpu") -> None:
        super(Horus,self).__init__()


        self.model = model_class()

        self.checkpoint = CheckPoint(model_file).load()

        self.state_dict = self.checkpoint.getStateDict()
        
        self.model.load_state_dict(self.state_dict)
        self.model.to(device)
        self.model.eval()
    
    def getStateDict(self):
        return self.state_dict
    
    def getCheckPoint(self) -> CheckPoint:
        return self.checkpoint

    def getModel(self) -> any:
        return self.model
    
    def predict(self,img:any) -> any:
        """
        Prediction of any image type based on saved model

        Args:
            img: any image type (`PIL`, `np.array` ...)
        """

        pred = self.model(img)
        return pred



class HorusModelTeacherSpatial(nn.Module):
    """
    Horus Teacher CNN Spatial Model.
    """
    def __init__(self):
        super(HorusModelTeacherSpatial,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    


class HorusModelTeacherTemporal(nn.Module):
    """
    Horus Teacher CNN Temporal Model.
    """
    def __init__(self):
        super(HorusModelTeacherTemporal,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class HorusModelStudentSpatial(nn.Module):
    """
    Horus Student CNN Spatial Model.
    """
    def __init__(self):
        super(HorusModelStudentSpatial, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x, spatiotemporal=False):
        
        x = self.encoder(x)
        if not spatiotemporal:
            x = self.decoder(x)

        return x


class HorusModelStudentTemporal(nn.Module):
    """
    Horus Student CNN Temporal Model.
    """
    def __init__(self):
        super(HorusModelStudentTemporal, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x, spatiotemporal=False):
        
        x = self.encoder(x)
        if not spatiotemporal:
            x = self.decoder(x)

        return x


def random_normal_fusion(img1:any, img2:any) -> any:
    """
    Random Normal Distribution implementing SpatioTemporal Fusion
    """
    weights = torch.randn_like(img1)
    weights = torch.clamp(weights, 0, 1)

    weights /= weights.sum(dim=1, keepdim=True)

    fused_img = (weights * img1) + ((1 - weights) * img2)

    return fused_img


class HorusSpatioTemporalModel:
    def __init__(self,classModelSpatial,classModelTemporal) -> None:
        self.spatialModel = classModelSpatial
        self.temporalModel = classModelTemporal

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        )
    
    def forward(self,x_s,x_t):

        x_s = self.spatialModel(x_s,spatiotemporal=True)
        x_t = self.temporalModel(x_t,spatiotemporal=True)

        x = random_normal_fusion(x_s,x_t)

        x = self.decoder(x)
        padding = (0, 3)
        x = F.pad(x, padding, mode='constant', value=0)

        return x