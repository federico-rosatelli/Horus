from dataLoader import dataLoader
from . import *
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp


train = dataLoader.newLoader(ROOT_DIR,TRAIN_SET)
valid = dataLoader.newLoader(ROOT_DIR,VALID_SET)
test = dataLoader.newLoader(ROOT_DIR,TEST_SET)


class Horus(nn.Module):
    def __init__(self):
        super(Horus, self).__init__()

        # Define the layers for your model
        self.conv16 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        
        self.max16  = nn.MaxPool2d(2,16)
        self.max32  = nn.MaxPool2d(2,32)

        self.conv32 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)


        self.conv64 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.conv128 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv16(x)
        x = self.relu(x)
        x = self.conv16(x)
        x = self.relu(x)

        x = self.max16(x)

        x = self.conv32(x)
        x = self.relu(x)

        x = self.max32(x)

        x = self.conv64(x)
        x = self.relu(x)
        x = self.conv64(x)
        x = self.relu(x)
        x = self.conv128(x)
        x = self.relu(x)
        return x

# Lsp = u * Ls * ()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Horus()
model.to(device)

