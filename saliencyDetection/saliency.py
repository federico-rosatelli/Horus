from matplotlib import pyplot as plt
from saliencyDetection.dataLoader import dataLoader as dtL
from . import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2

import logging

logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False



train = dtL.newLoader(ROOT_DIR,TRAIN_SET)
valid = dtL.newLoader(ROOT_DIR,VALID_SET)
test = dtL.newLoader(ROOT_DIR,TEST_SET)

trainT = dtL.newTeacherLoader(ROOT_DIR,TRAIN_SET)
validT = dtL.newTeacherLoader(ROOT_DIR,VALID_SET)
testT = dtL.newTeacherLoader(ROOT_DIR,TEST_SET)


class HorusModelTeacher(nn.Module):
    def __init__(self):
        super(HorusModelTeacher,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 1280, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x






class HorusModel(nn.Module):
    def __init__(self):
        super(HorusModel, self).__init__()

        # Define the layers for your model
        self.conv16 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv256_16 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        self.max16  = nn.MaxPool2d(1,1)
        self.max32  = nn.MaxPool2d(1,1)

        self.conv32 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)


        self.conv64 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv64_64 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv128 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv256 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.convP = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.max256  = nn.MaxPool2d((1,2),(1,2))


    def forward(self, x):
        print(x.size)
        x = self.conv16(x)
        x = self.relu(x)
        x = self.conv256_16(x)
        

        
        x = self.max16(x)

        

        x = self.conv32(x)


        x = self.max32(x)


        x = self.conv64(x)
        x = self.relu(x)
        x = self.conv64_64(x)
        x = self.relu(x)
        x = self.conv128(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.conv256(x)
        #print(x.shape)
        x = self.relu(x)

        return x

# Lsp = u * Ls * ()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Horus()
# model.to(device)
    
class Horus:
    HOME_PATH:str = "saliencyDetection/"
    model:HorusModel = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self,model_file:str="horus_model.pt",state_dict:bool=False) -> None:
        super(Horus,self).__init__()

        model_file = self.HOME_PATH+model_file

        if state_dict:
            self.model = HorusModel()
            self.model.load_state_dict(torch.load(model_file))
        else:
            self.model = torch.load(model_file)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self,img:any) -> np.ndarray:
        img = img.to(self.device)
        pred = self.model(img)
        return pred.detach().cpu().numpy()


def trainTeacher(epoch:int=10,verbose:str|None=None):
    if verbose:
        logger = logging.getLogger(verbose)
    
    model = HorusModelTeacher()
    device = torch.device("cpu")
    model = model.to(device)

    learning_rate = 0.0001
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)

    model.train()
    for k in range(epoch):
        for img,label in trainT:
            x = img.to(device)
            y = label.to(device)

            predict = model(x)
            print(predict.size(),y.size())
            loss = loss_fn(predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            #logger.info(loss)
            if loss>0.02:
                    if verbose:
                        logger.info(f"Loss: {loss}")
                    show(predict,y)


def trainHorus(epoch:int=10,verbose:str|None=None):
    if verbose:
        logger = logging.getLogger(verbose)
    model = HorusModel()
    device = torch.device("cpu")
    model = model.to(device)

    learning_rate = 0.0001
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    min_loss = 0
    model.train()
    for k in range(epoch):
        for img,label,name in train:
            mean = 0
            #print(len(img),name)
            for i in range(len(img)-1):
                
                x_spatial = img[i]
                y_spatial = label[i]
                x_temporal = [img[i],img[i+1]]
                y_temporal = [label[i],label[i+1]]
                #print(x_spatial.shape)
                x_spatial = x_spatial.to(device)
                y_spatial = y_spatial.to(device)
                pre_spatial = model(x_spatial)

                loss = loss_fn(pre_spatial,y_spatial)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()
                mean += loss
                
                if loss>0.02:
                    if verbose:
                        logger.info(f"loss saved: {loss} for {name[0]}")
                    show(pre_spatial,y_spatial,img[i])
                    #print("QUANTI 0? ",np.count_nonzero(pre_spatial==0),np.count_nonzero(pre_spatial!=0))
                if loss > min_loss:
                    min_loss = loss
                    if verbose:
                        logger.info(f"{loss} in: {i}")
                    
                    # if loss>0.03:
                    #     show(pre_spatial,y_spatial,loss)
               


            
    # for k in range(epoch):
    #     mean = 0
    #     print(f"{k} EPOCH")
    #     for i in range(len(train)):
    #         for x,y,z in train:
    #             # for i in range(len(x)):
    #             #     x[i] = x[i].to(device)
    #             #     y[i] = y[i].to(device)
    #             x = x.to(device)
    #             y = y.to(device)

    #             pred = model(x)
    #             #print(pred.shape,y.shape)
    #             loss = loss_fn(pred,y)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             loss = loss.item()
    #             mean += loss
    #             if loss > min_loss:
    #                 min_loss = loss
    #                 print(f"{loss} in: {i}")
    #                 print(f"mean loss: {mean/(i+1)}")
    #                 # if loss>0.06:
    #                 #     show(pred,y)
        if verbose:
            logger.warning(f"mean loss: {mean/len(train)}")




def show(img,label)->None:
    
    # # print(img)
    # #image = (255.0 * img).to(torch.uint8)
    # print(type(img))
    #print(img.size(),label.size())
    image = (img*255).detach().cpu().numpy().transpose(0, 2, 1, 3)
    # # print(image.shape)
    # image = image[:, :, :, ::-1]
    # print(image.shape)
    #cv2.imwrite("testss/img/test.png",image)


    #np.save("testss/img/test.png", img.detach().cpu().numpy())
    fig, axarr = plt.subplots(1,2)
    
    grayscale_transform = transforms.Grayscale(num_output_channels=1)  # Specify 1 channel for grayscale
    grayscale_tensor = grayscale_transform(torch.from_numpy(np.array(image)).float())
    #grayscale_tensor = torch.from_numpy(np.array(image)).float().squeeze(0)
    
    axarr[0].imshow(grayscale_tensor.detach().numpy().reshape(720,1280,1))
    axarr[0].set_title('Image')
    axarr[0].axis('off')

    axarr[1].imshow((label*255).detach().numpy().reshape(256,256,1))
    axarr[1].set_title('Label')
    axarr[1].axis('off')

    # axarr[2].imshow((img1).detach().numpy().reshape(256,256,-1))
    # axarr[2].set_title('Original Img')
    # axarr[2].axis('off')

    #fig.suptitle(f'Image & Label of {self.type}/{self.item} on Frame:{self.nframe}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"testss/img/test.png")
    plt.close(fig)



