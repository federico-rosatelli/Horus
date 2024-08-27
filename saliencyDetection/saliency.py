# AUTHOR Federico Rosatelli

"""
`saliencyDetection.saliency` is a Horus module that implements and manages the entire 
`Horus` neural network, from training to its use.
"""
import logging
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False
from matplotlib import pyplot as plt
from saliencyDetection.dataLoader import dataLoader as dtL
import saliencyDetection.lossFunction as lossFunction
from . import *
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F



class HorusModelTeacher(nn.Module):
    """
    Horus Teacher CNN Model.
    """
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






class HorusModelStudent(nn.Module):
    """
    Horus Student CNN Model.
    """
    def __init__(self):
        super(HorusModelStudent, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

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


    def forward(self, x, spatiotemporal=False):
        
        x = self.encoder(x)
        if not spatiotemporal:
            x = self.decoder(x)
            padding = (0, 3)
            x = F.pad(x, padding, mode='constant', value=0)
        return x


    
class Horus:
    """
    Horus Class Prediction.

    Args:
        model_class: any Model Class.
        model_file: string, name of the model file saved (eg. `pt` files).
        state_dict: boolean, if the model file is a state dictionary.
    """
    HOME_PATH:str = f"{DIR}/models/"
    device = torch.device('cpu')
    def __init__(self,model_class:any,model_file:str="horus_model.pt",state_dict:bool=True) -> None:
        super(Horus,self).__init__()

        model_file = self.HOME_PATH+model_file

        if state_dict:
            self.model = model_class()
            self.model.load_state_dict(torch.load(model_file))
        else:
            self.model = torch.load(model_file)
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self,img:any) -> any:
        """
        Prediction of any image type based on saved model

        Args:
            img: any image type (`PIL`, `np.array` ...)
        """

        pred = self.model(img)
        return pred




def trainHorusNetwork(conf:any,verbose:str|None=None) -> None:
    """
    Start the training of Horus Neural Network.

    The function implements both teacher and student training.
    
    If any Model File is found in the models path it'll skip the training of that model.

    Args:
        conf: a configuration dictionary from config path.
        verbose: a string for :class:`logging` module to return a logger with the specified name.
    """

    teacherConf = conf["teacher"]["training"]
    studentConf = conf
    
    pathTeacher = Path(f'{DIR}/models/{teacherConf["files"]["ModelSpatial"]}')  #teacher model
    if not pathTeacher.exists():                                                #skip if model exists

        # Spatial/Temporal Teacher Training
        trainHorusTeacher(teacherConf,dtL.AVS1KDataSetTeacher,HorusModelTeacher,verbose)

    pathStudent = Path(f'{DIR}/models/{studentConf["files"]["ModelSpatial"]}')  #student model
    if not pathStudent.exists():                                                #skip if model exists

        # Spatial/Temporal Student Training
        trainHorusStudent(studentConf,dtL.AVS1KDataSet,HorusModelStudent,verbose=verbose)
    
    #TODO Spatial Temporal Model Training (???)






def trainHorusTeacher(conf:any,datasetCLass:any,model:any,verbose:str|None=None) -> None:
    """
    Training of Horus Teacher Model.

    Spatial & Temporal Knowledge implemented.

    Args:
        conf: a configuration dictionary from config path.
        datasetClass: Horus DataLoader class from :file:`dataLoader` module.
        model: Horus Model Class.
        verbose: a string for :class:`logging` module to return a logger with the specified name.
    """

    files = conf["files"]
    
    if verbose:
        logger = logging.getLogger(verbose)
    
    epochs = int(conf["epochs"])
    if epochs < 0 or epochs > 4096:
        raise ValueError(f"Value for build must be > 0 & < 4097 not {epochs}")
    
    batch_size = int(conf["batch_size"])
    if batch_size < 0:
        raise ValueError(f"Value for batch must be > 0. Got: {batch_size}")
    
    batch_saving = int(conf["batch_saving"])
    if batch_saving < 0:
        raise ValueError(f"Value for batch_saving must be > 0. Got: {batch_saving}")
        
    train = dtL.newLoader(datasetCLass,ROOT_DIR,TRAIN_SET,batch_size)

    device = torch.device("cpu")

    modelS = model()            #spatial model
    modelT = model()            #temporal model

    learning_rate = conf["learning_rate"]

    loss_fn_S = nn.MSELoss()
    loss_fn_T = nn.MSELoss()

    optimizerS = torch.optim.AdamW(modelS.parameters(),learning_rate)
    optimizerT = torch.optim.AdamW(modelT.parameters(),learning_rate)

    epochs_historyS = []
    epochs_historyT = []

    modelS.train()
    modelT.train()

    min_loss_S = 1
    min_loss_T = 2

    avg_loss_s = 1
    avg_loss_t = 2

    for k in range(epochs):
        if verbose:
            logger.info(f"TEACHER TRAINING EPOCH: {k+1} of {epochs}")
        batch_historyS = []
        batch_historyT = []
        for batch,(imgs,labels) in enumerate(train):
            
            x_spatial,x_temporal = imgs
            y_spatial,y_temporal = labels

            x_spatial = x_spatial.to(device)
            x_temporal = x_temporal.to(device)

            y_spatial = y_spatial.to(device)
            y_temporal = y_temporal.to(device)

            x_temporal = torch.cat((x_spatial,x_temporal))
            y_temporal = torch.cat((y_spatial,y_temporal))

            optimizerS.zero_grad()
            optimizerT.zero_grad()

            predictS = modelS(x_spatial)
            predictT = modelT(x_temporal)

            loss_S = loss_fn_S(predictS,y_spatial)
            loss_T = loss_fn_T(predictT,y_temporal)

            loss_S.backward()
            loss_T.backward()

            optimizerS.step()
            optimizerT.step()

            loss_S = loss_S.item()
            loss_T = loss_T.item()

            batch_historyS.append(loss_S)
            batch_historyT.append(loss_T)

            if batch % batch_saving == 0 and batch != 0:

                latest_batch_s = batch_historyS[-batch_saving:]
                latest_batch_t = batch_historyT[-batch_saving:]

                avg_s = sum(latest_batch_s)/batch_saving
                min_s = min(latest_batch_s)

                avg_t = sum(latest_batch_t)/batch_saving
                min_t = min(latest_batch_t)


                if verbose:
                    logger.info(f"Avg Spatial Loss Batch {batch}: {avg_s} ¦ Min Loss: {min_s}")
                    logger.info(f"Avg Temporal Loss Batch {batch}: {avg_t} ¦ Min Loss: {min_t}")

                if avg_s < avg_loss_s:
                    if verbose:
                        logger.info(f"Saving Spatial Model...")
                    torch.save(modelS.state_dict(), f"{DIR}/models/{files['ModelSpatial']}")
                    avg_loss_s = avg_s
                
                if avg_t < avg_loss_t:
                    if verbose:
                        logger.info(f"Saving Temporal Model...")
                    torch.save(modelT.state_dict(), f"{DIR}/models/{files['ModelTemporal']}")
                    avg_loss_t = avg_t
            
            if loss_S < min_loss_S:
                if verbose:
                    logger.info(f"Saving Spatial Model for min Loss: {loss_S}...")
                torch.save(modelS.state_dict(), f"{DIR}/models/{files['ModelSpatial']}")
                min_loss_S = loss_S

            if loss_T < min_loss_T:
                if verbose:
                    logger.info(f"Saving Temporal Model for min Loss: {loss_T}...")
                torch.save(modelT.state_dict(), f"{DIR}/models/{files['ModelTemporal']}")
                min_loss_T = loss_T


        epochs_historyS.append(batch_historyS)
        epochs_historyT.append(batch_historyT)

        batch_historyS = []
        batch_historyT = []
    
    history_dictS = {}
    history_dictT = {}

    for i in range(len(batch_historyS)):
        history_dictS[i] = batch_historyS[i]
        history_dictT[i] = batch_historyT[i]
    
    with open(f"{DIR}/json/{files['LossHistorySpatial']}","w") as jswr:
        json.dump(history_dictS,jswr)
    
    with open(f"{DIR}/json/{files['LossHistoryTemporal']}","w") as jswr:
        json.dump(history_dictT,jswr)


    return


def trainHorusStudent(conf:any,datasetClass:any,model:any,verbose:str|None=None) -> None:
    """
    Training of Horus Student Model.\n
    Spatial & Temporal Knowledge implemented.

    Args:
        conf: a configuration dictionary from config path.
        datasetClass: Horus DataLoader class from :module:`saliencyDetection.dataLoader.dataLoader` module.
        model: Horus Model Class.
        verbose: a string for :class:`logging` module to return a logger with the specified name.
    """


    models = conf["teacher"]["training"]["files"]
    conf = conf["student"]["training"]
    files = conf["files"]
    
    if verbose:
        logger = logging.getLogger(verbose)
    
    teacherModelS = Horus(HorusModelTeacher,models["ModelSpatial"],state_dict=True)       #teacher spatial model already trained
    teacherModelT = Horus(HorusModelTeacher,models["ModelTemporal"],state_dict=True)      #teacher temporal model already trained

    epochs = int(conf["epochs"])
    if epochs < 0 or epochs > 4096:
        raise ValueError(f"Value for build must be > 0 & < 4097 not {epochs}")
    
    batch_size = int(conf["batch_size"])
    if batch_size < 0:
        raise ValueError(f"Value for batch must be > 0. Got: {batch_size}")
    
    batch_saving = int(conf["batch_saving"])
    if batch_saving < 0:
        raise ValueError(f"Value for batch_saving must be > 0. Got: {batch_saving}")
    
    train = dtL.newLoader(datasetClass,ROOT_DIR,TRAIN_SET,batch_size)
    
    device = torch.device("cpu")
    
    modelT = model()           # temporal model for student
    modelS = model()           # spatial model for student
    modelS = modelS.to(device)
    modelT = modelT.to(device)


    learning_rate = conf["learning_rate"]
    
    #loss_fn = nn.MSELoss()
    
    my_loss_fn_s = lossFunction.HorusLossFunction()   # my custom loss function spatial
    my_loss_fn_t = lossFunction.HorusLossFunction()   # my custom loss function temporal

    optimizerS = torch.optim.AdamW(modelS.parameters(),learning_rate)   #optimizer for spatial model
    optimizerT = torch.optim.AdamW(modelT.parameters(),learning_rate)   #optimizer for temporal model

    modelT.train()
    modelS.train()

    epochs_historyS = []
    epochs_historyT = []

    min_loss_S = 1
    min_loss_T = 1

    avg_loss_s = 1
    avg_loss_t = 1

    

    for k in range(epochs):
        batch_historyS = []
        batch_historyT = []

        if verbose:
            logger.info(f"STUDENT TRAINING EPOCH: {k+1} of {epochs}")
        for batch,(imgs,labels) in enumerate(train):
            
            teacher_x_st,student_x_st = imgs
            teacher_y_st,student_y_st = labels

            teacher_x_s = teacher_x_st[0].to(device)            #spatial image  teacher
            teacher_x_t = teacher_x_st[1].to(device)            #temporal image teacher

            student_x_s = student_x_st[0].to(device)            #spatial image  student
            student_x_t = student_x_st[1].to(device)            #temporal image student

            teacher_y_s = teacher_y_st[0].to(device)            #spatial label  teacher
            teacher_y_t = teacher_y_st[1].to(device)            #temporal label teacher

            student_y_s = student_y_st[0].to(device)            #spatial label  student
            student_y_t = student_y_st[1].to(device)            #temporal label student

            
            teacher_x_t = torch.cat((teacher_x_s,teacher_x_t))  #create temporal teacher image stack
            teacher_y_t = torch.cat((teacher_y_s,teacher_y_t))  #create temporal teacher label stack

            student_x_t = torch.cat((student_x_s,student_x_t))  #create temporal student image stack
            student_y_t = torch.cat((student_y_s,student_y_t))  #create temporal studente label stack
            
            optimizerT.zero_grad()
            optimizerS.zero_grad()


            predictS = modelS(student_x_s)                              #spatial student prediction
            predictT = modelT(student_x_t)                              #temporal student prediction

            teacherPredS = teacherModelS.predict(teacher_x_s)           #spatial teacher prediction
            teacherPredT = teacherModelT.predict(teacher_x_t)           #temporal teacher prediction

            #we're going to apply the teacher prediction on the custom loss function

            loss_S = my_loss_fn_s(predictS,teacherPredS,teacher_y_s)    #spatial loss
            loss_T = my_loss_fn_t(predictT,teacherPredT,teacher_y_t)    #temporal loss

            loss_S.backward()
            loss_T.backward()

            optimizerS.step()
            optimizerT.step()

            loss_S = loss_S.item()
            loss_T = loss_T.item()

            if batch % batch_saving == 0 and batch != 0:
                
                latest_batch_s = batch_historyS[-batch_saving:]
                latest_batch_t = batch_historyT[-batch_saving:]

                avg_s = sum(latest_batch_s)/batch_saving            #average spatial loss of batch length
                min_s = min(latest_batch_s)                         #min spatial loss of batch length

                avg_t = sum(latest_batch_t)/batch_saving            #average temporal loss of batch length
                min_t = min(latest_batch_t)                         #min temporal loss of batch length


                if verbose:
                    logger.info(f"Avg Spatial Loss Batch {batch}: {avg_s} ¦ Min Loss: {min_s}")
                    logger.info(f"Avg Temporal Loss Batch {batch}: {avg_t} ¦ Min Loss: {min_t}")

                if avg_s < avg_loss_s:      #if the average spatial loss is less then the minimum average spatial loss
                    if verbose:
                        logger.info(f"Saving Spatial Model...")
                    torch.save(modelS.state_dict(), f"{DIR}/models/{files['ModelSpatial']}")    #save the model
                    avg_loss_s = avg_s
                
                if avg_t < avg_loss_t:      #if the average temporal loss is less then the minimum average temporal loss
                    if verbose:
                        logger.info(f"Saving Temporal Model...")
                    torch.save(modelT.state_dict(), f"{DIR}/models/{files['ModelTemporal']}")   #save the model
                    avg_loss_t = avg_t
            
            if loss_S < min_loss_S:         #if the spatial loss is less then the minumum spatial loss
                if verbose:
                    logger.info(f"Saving Spatial Model for min Loss: {loss_S}...")
                torch.save(modelS.state_dict(), f"{DIR}/models/{files['ModelSpatial']}")        #save the model
                min_loss_S = loss_S

            if loss_T < min_loss_T:         #if the temporal loss is less then the minumum temporal loss
                if verbose:
                    logger.info(f"Saving Temporal Model for min Loss: {loss_T}...")
                torch.save(modelT.state_dict(), f"{DIR}/models/{files['ModelTemporal']}")       #save the model
                min_loss_T = loss_T


        epochs_historyS.append(batch_historyS)
        epochs_historyT.append(batch_historyT)

        batch_historyS = []
        batch_historyT = []
    
    history_dictS = {}
    history_dictT = {}

    for i in range(len(batch_historyS)):
        history_dictS[i] = batch_historyS[i]
        history_dictT[i] = batch_historyT[i]
    
    with open(f"{DIR}/json/{files['LossHistorySpatial']}","w") as jswr:     #save spatial loss history in json file
        json.dump(history_dictS,jswr)
    
    with open(f"{DIR}/json/{files['LossHistoryTemporal']}","w") as jswr:    #save temporal loss history in json file
        json.dump(history_dictT,jswr)

    # TO spatiotemporalTraining
    return


###TODO spatiotemporal training and model (??????)
#random normal distribution
def random_normal_fusion(img1, img2) -> any:
  
    weights = torch.randn_like(img1)
    weights = torch.clamp(weights, 0, 1)

    weights /= weights.sum(dim=1, keepdim=True)

    fused_img = (weights * img1) + ((1 - weights) * img2)

    return fused_img


class HorusSpatioTemporalModel:
    def __init__(self,classModelSpatial,classModelTemporal) -> None:
        self.spatial = classModelSpatial
        self.temporal = classModelTemporal
        # qui vanno messi i layers
    
    def forward(self,x_s,x_t):

        x_s = self.spatial(x_s)
        x_t = self.temporal(x_t)

        x = random_normal_fusion(x_s,x_t)

        # layers della cnn

        return x




