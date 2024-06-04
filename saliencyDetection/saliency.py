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




train = dtL.newLoader(ROOT_DIR,TRAIN_SET)
valid = dtL.newLoader(ROOT_DIR,VALID_SET)
test = dtL.newLoader(ROOT_DIR,TEST_SET)


class Horus(nn.Module):
    def __init__(self):
        super(Horus, self).__init__()

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
        
        x = self.conv16(x)
        x = self.relu(x)
        x = self.conv256_16(x)
        
        #x = self.relu(x)
        #print(x.shape)
        x = self.max16(x)
        #print(x.shape)

        

        x = self.conv32(x)
        #x = self.relu(x)

        #print(x.shape)
        x = x.unsqueeze(3)
        x = self.max32(x)
        x = x.squeeze(3)
        #print(x.shape)

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
        #x = x.reshape()

        # x = x.unsqueeze(2)
        # print(x.shape)
        # #x = self.upsample_layer(x)
        # x = self.max256(x)
        # x = x.squeeze(2)
        return x

# Lsp = u * Ls * ()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Horus()
# model.to(device)
    
def trainHorus(epoch):
    model = Horus()
    device = torch.device("cpu")
    model = model.to(device)

    learning_rate = 0.001
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    min_loss = 0
    model.train()
    print(len(train))
    for k in range(epoch):
        mean = 0
        print(f"{k} EPOCH")
        for i in range(len(train)):
            for x,y,z in train:
                # for i in range(len(x)):
                #     x[i] = x[i].to(device)
                #     y[i] = y[i].to(device)
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                #print(pred.shape,y.shape)
                loss = loss_fn(pred,y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = loss.item()
                mean += loss
                if loss > min_loss:
                    min_loss = loss
                    print(f"{loss} in: {i}")
                    print(f"mean loss: {mean/(i+1)}")
                    # if loss>0.06:
                    #     show(pred,y)

        print(f"mean loss: {mean/len(train)}")



def show(img,label)->None:
    fig, axarr = plt.subplots(1,2)
    print(type(img))
    axarr[0].imshow(img.detach().numpy().reshape(256,256,3))
    axarr[0].set_title('Image')
    axarr[0].axis('off')

    axarr[1].imshow(label.permute(2,0,1))
    axarr[1].set_title('Label')
    axarr[1].axis('off')

    #fig.suptitle(f'Image & Label of {self.type}/{self.item} on Frame:{self.nframe}', fontsize=10)
    plt.tight_layout()
    plt.show()



