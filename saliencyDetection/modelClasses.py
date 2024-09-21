import logging
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False
from . import *
import torch.nn as nn
import torch


class Horus:
    """
    Horus Class Prediction.

    Args:
        model_class: any Model Class.
        model_file: string, name of the model file saved (eg. `pt` files).
        state_dict: boolean, if the model file is a state dictionary.
        device: string, `cpu`|`cuda` from the config file
    """
    HOME_PATH:str = f"{DIR}/models/"
    
    def __init__(self,model_class:any,model_file:str="horus_model.pt",state_dict:bool=True,device:str="cpu") -> None:
        super(Horus,self).__init__()

        model_file = self.HOME_PATH+model_file

        if state_dict:
            self.model = model_class()
            self.model.load_state_dict(torch.load(model_file))
        else:
            self.model = torch.load(model_file)
        
        self.model.to(device)
        self.model.eval()
    
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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
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
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=1, stride=2, padding=1, output_padding=1),
            nn.ReLU()
            
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
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, spatiotemporal=False):
        
        x = self.encoder(x)
        if not spatiotemporal:
            x = self.decoder(x)

        return x


class HorusModelStudentTemporal(nn.Module):
    """
    Horus Student CNN Model.
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

