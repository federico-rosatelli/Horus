from matplotlib import pyplot as plt
from saliencyDetection.dataLoader import dataLoader as dtL
import saliencyDetection.lossFunction as lossFunction
from . import *
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import logging

logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False



trainS = dtL.newLoader(dtL.AVS1KDataSetStudent,ROOT_DIR,TRAIN_SET)
validS = dtL.newLoader(dtL.AVS1KDataSetStudent,ROOT_DIR,VALID_SET)
testS = dtL.newLoader(dtL.AVS1KDataSetStudent,ROOT_DIR,TEST_SET)

trainT = dtL.newLoader(dtL.AVS1KDataSetTeacher,ROOT_DIR,TRAIN_SET)
validT = dtL.newLoader(dtL.AVS1KDataSetTeacher,ROOT_DIR,VALID_SET)
testT = dtL.newLoader(dtL.AVS1KDataSetTeacher,ROOT_DIR,TEST_SET)

train = dtL.newLoader(dtL.AVS1KDataSet,ROOT_DIR,TRAIN_SET)
valid = dtL.newLoader(dtL.AVS1KDataSet,ROOT_DIR,VALID_SET)
test = dtL.newLoader(dtL.AVS1KDataSet,ROOT_DIR,TEST_SET)

class HorusModelTeacher(nn.Module):
    def __init__(self):
        super(HorusModelTeacher,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(720, 256, kernel_size=3, padding=1),
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
            nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 720, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        padding = (0, 3)
        x = F.pad(x, padding, mode='constant', value=0)
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
    HOME_PATH:str = f"{DIR}/"
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
        """
        Prediction of any image type based on saved model
        """
        img = img.to(self.device)
        pred = self.model(img)
        return pred.detach().cpu().numpy()


def trainTeacher(conf:any,verbose:str|None=None):
    files = conf["files"]
    
    if os.path.isfile(f"{DIR}/{files['Model']}"):
        os.remove(f"{DIR}/{files['Model']}")
    if verbose:
        logger = logging.getLogger(verbose)
    
    
    config = conf["training"]
    epochs = int(config["epochs"])
    if epochs < 0 or epochs > 4096:
        raise ValueError(f"Value for build must be > 0 & < 4097 not {epochs}")
    
    batch_size = int(config["batch_size"])
    if batch_size < 0 or batch_size > len(trainT):
        raise ValueError(f"Value for build must be > 0 & < {len(trainT)} not {batch_size}")

    model = HorusModelTeacher()
    device = torch.device("cpu")
    model = model.to(device)

    learning_rate = float(config["learning_rate"])
    my_loss_fn = lossFunction.HorusLossFunction()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(),learning_rate)

    model.train()

    #history = []
    mean_history = []
    min_mean_batch_loss = 1
    min_loss = 1
    for k in range(epochs):
        batch_history = []
        logger.info(f"EPOCH: {k+1}")
        for batch,(img,label) in enumerate(trainT):
            x = img.to(device)
            y = label.to(device)
            
            optimizer.zero_grad()
            predict = model(x)
            loss = loss_fn(predict,y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            #logger.info(loss)
            #history.append(loss)
            batch_history.append((predict,y,loss))
            if batch % batch_size == 0 and batch != 0:
                #mean_loss = sum(history)/len(history)
                mean_loss_batch = sum([z[2] for z in batch_history])/len(batch_history)
                
                if verbose:
                    logger.info(f"Mean Batch Loss: {mean_loss_batch}. {batch} to {len(trainT)}")
                
                mean_history.append(mean_loss_batch)
                min_batch_loss = min(batch_history,key=lambda z:z[2])
                
                show(min_batch_loss[0],min_batch_loss[1],min_batch_loss[2],"test.png")
                batch_history = []
                if sum(mean_history)/len(mean_history) >= mean_loss_batch:
                    #min_mean_batch_loss = mean_loss_batch
                    torch.save(model.state_dict(), f"{DIR}/{files['Model']}")
                    if verbose:
                        logger.info(f"Saving batch {batch} in model")
                    

            if loss<min_loss:
                min_loss = loss
                logger.info(f"Min Loss: {min_loss} at {batch}")
                show(predict,y,loss,"test_min.png")
    json_dict = {
        "mean_history":mean_history
    }
    with open(f"{DIR}/{files['JSONFormat']}","w") as out:
        json.dump(json_dict,out,indent=4)


def trainHorus(conf:any,verbose:str|None=None):
    files = conf["files"]
    
    # if os.path.isfile(f"{DIR}/{files['Model']}"):
    #     os.remove(f"{DIR}/{files['Model']}")
    if verbose:
        logger = logging.getLogger(verbose)
    
    
    config = conf["training"]
    epochs = int(config["epochs"])
    if epochs < 0 or epochs > 4096:
        raise ValueError(f"Value for build must be > 0 & < 4097 not {epochs}")
    
    batch_size = int(config["batch_size"])
    if batch_size < 0 or batch_size > len(trainT):
        raise ValueError(f"Value for build must be > 0 & < {len(trainT)} not {batch_size}")
    
    device = torch.device("cpu")
    
    modelT = HorusModelTeacher()    # model for teacher
    modelS = HorusModel()           # model for student
    modelT = modelT.to(device)
    modelS = modelS.to(device)


    learning_rateT = config["teacher"]["learning_rate"]
    learning_rateS = config["student"]["learning_rate"]
    #loss_fn = nn.MSELoss()

    my_loss_fn = lossFunction.HorusLossFunction()   # my custom loss function

    optimizerT = torch.optim.Adam(modelT.parameters(),learning_rateT)
    optimizerS = torch.optim.Adam(modelS.parameters(),learning_rateS)
    mean_history = []
    batch_history = []
    modelT.train()
    modelS.train()
    min_loss = 1
    for k in range(epochs):
        if verbose:
            logger.info(f"EPOCH: {k+1} of {epochs}")
        for batch,(imgs,labels) in enumerate(train):
            
            teacher_x_st,student_x_st = imgs
            teacher_y_st,student_y_st = labels

            teacher_x_s = teacher_x_st[0].to(device)
            teacher_x_t = teacher_x_st[1].to(device)

            student_x_s = student_x_st[0].to(device)
            student_x_t = student_x_st[1].to(device)

            teacher_y_s = teacher_y_st[0].to(device)
            teacher_y_t = teacher_y_st[1].to(device)

            student_y_s = student_y_st[0].to(device)
            student_y_t = student_y_st[1].to(device)

            
            optimizerT.zero_grad()
            optimizerS.zero_grad()

            predictT = modelT(teacher_x_s)
            predictS = modelS(student_x_s)

            loss = my_loss_fn(predictT,predictS,teacher_y_s)
            loss.backward()

            optimizerT.step()
            optimizerS.step()

            loss = loss.item()

            batch_history.append(loss)

            if batch % batch_size == 0 and batch != 0:
                
                mean_loss_batch = sum(batch_history)/len(batch_history)
                mean_history.append(mean_loss_batch)
                min_batch_loss = min(batch_history)

                lasts_loss = mean_history[-abs(len(mean_history)*0.2):]
                mean_lasts_loss = sum(lasts_loss)/len(lasts_loss)

                if verbose:
                    logger.info(f"Mean Batch Loss: {mean_loss_batch} ¦ Min Loss: {min_batch_loss} ¦ {batch} to {len(trainT)}")
                
                if mean_loss_batch <= mean_lasts_loss:
                    torch.save(modelS.state_dict(), f"{DIR}/{files['Model']}")
                    if verbose:
                        logger.info(f"Saving batch {batch} in model")
            
            if loss<min_loss:
                min_loss = loss
                logger.info(f"Min Loss: {min_loss} at {batch}")
    #         batch_history.append((predict,spatial_y,loss))
    #         if batch % batch_size == 0 and batch != 0:
                
    #             mean_loss_batch = sum([z[2] for z in batch_history])/len(batch_history)
                
    #             if verbose:
    #                 logger.info(f"Mean Batch Loss: {mean_loss_batch}. {batch} to {len(trainT)}")
                
    #             mean_history.append(mean_loss_batch)
    #             min_batch_loss = min(batch_history,key=lambda z:z[2])
                
    #             show(min_batch_loss[0],min_batch_loss[1],min_batch_loss[2],"test_studente.png",(256,256,1))
    #             batch_history = []
    #             if sum(mean_history)/len(mean_history) >= mean_loss_batch:
    #                 #min_mean_batch_loss = mean_loss_batch
    #                 torch.save(model.state_dict(), f"{DIR}/{files['Model']}")
    #                 if verbose:
    #                     logger.info(f"Saving batch {batch} in model")
                    

    #         if loss<min_loss:
    #             min_loss = loss
    #             logger.info(f"Min Loss: {min_loss} at {batch}")
    #             show(predict,spatial_y,loss,"test_min_student.png",(256,256,1))
    # json_dict = {
    #     "mean_history":mean_history
    # }
    # with open(f"{DIR}/{files['JSONFormat']}","w") as out:
    #     json.dump(json_dict,out,indent=4)



def show(img,label,min_loss,file_name,size=(720,1280,1))->None:
    
    
    fig, axarr = plt.subplots(1,2)
    if size == (256,256,1):
        
        img = (img*255).detach().cpu().numpy().transpose(0, 2, 1, 3)
        
        grayscale_transform = transforms.Grayscale(num_output_channels=1)  # Specify 1 channel for grayscale
        img = grayscale_transform(torch.from_numpy(np.array(img)).float())
        img = torch.from_numpy(np.array(img)).float().squeeze(0)
    
    axarr[0].imshow((img*255).detach().numpy().reshape(size[0],size[1],size[2]))
    axarr[0].set_title('Image')
    axarr[0].axis('off')

    axarr[1].imshow((label*255).detach().numpy().reshape(size[0],size[1],size[2]))
    axarr[1].set_title('Label')
    axarr[1].axis('off')

    
    plt.tight_layout()
    plt.title(f"Min Loss: {min_loss}")
    plt.savefig(f"testss/img/{file_name}")
    plt.clf()
    plt.close("all")
    plt.close(fig)
    plt.ioff()



