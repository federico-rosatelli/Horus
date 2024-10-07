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
from saliencyDetection import modelClasses
from saliencyDetection import training
from .utils import displayer, model
from . import *
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    studentConf = conf["student"]["training"]
    
    pathTeacherSpatial = Path(f'{DIR}/{MODEL_DIR}/{teacherConf["files"]["ModelSpatial"]}')

    if not pathTeacherSpatial.exists():                                                     #skip if model exists
        # Spatial Teacher Training
        training.trainHorusTeacher(conf,dtL.AVS1KDataSetTeacherSpatial,modelClasses.HorusModelTeacherSpatial,type_run="spatial",verbose=verbose)
        
    else:
        # Get CheckPoint Model and Start from that
        checkpoint = model.CheckPoint(teacherConf["files"]["ModelSpatial"]).load()
        if checkpoint.getEpoch() < teacherConf["epochs"]:
            newConf = checkpoint.exportConf(device=conf["device"],
                                            dataset_class=dtL.AVS1KDataSetTeacherSpatial,
                                            model_class=modelClasses.HorusModelTeacherSpatial, 
                                            epochs=teacherConf["epochs"],
                                            batch_size=teacherConf["batch_size"],
                                            learning_rate=teacherConf["learning_rate"],
                                            loss_function=lossFunction.getLossFunction(teacherConf["loss_function"]),
                                            batch_saving=teacherConf["batch_saving"])
            
            training.trainHorusTeacher_checkpoint(newConf,verbose=verbose)
            
    
    
    pathTeacherTemporal = Path(f'{DIR}/{MODEL_DIR}/{teacherConf["files"]["ModelTemporal"]}')
    
    if not pathTeacherTemporal.exists():                                                    #skip if model exists
        # Temporal Teacher Training    
        training.trainHorusTeacher(conf,dtL.AVS1KDataSetTeacherTemporal,modelClasses.HorusModelTeacherTemporal,type_run="temporal",verbose=verbose)

    pathStudent = Path(f'{DIR}/{MODEL_DIR}/{studentConf["files"]["ModelSpatial"]}')  #student model
    if not pathStudent.exists():                                                #skip if model exists

        teacherModelS = modelClasses.Horus(dtL.AVS1KDataSetTeacherSpatial,teacherConf["files"]["ModelSpatial"],state_dict=True,device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentSpatial,modelClasses.HorusModelStudentSpatial,teacherModelS,type_run="spatial",verbose=verbose)

        teacherModelT = modelClasses.Horus(dtL.AVS1KDataSetTeacherTemporal,teacherConf["files"]["ModelTemporal"],state_dict=True,device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentTemporal,modelClasses.HorusModelStudentTemporal,teacherModelT,type_run="temporal",verbose=verbose)
        
    
    #TODO Spatial Temporal Model Training (???)



###TODO spatiotemporal training and model (??????)
#random normal distribution
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


def trainSpatioTemporalHorus(conf:any,datasetClass:any,model:any,spatialModel:any,temporalModel:any,verbose:str|None=None) -> None:

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
    
    train = dtL.newLoader(datasetClass,ROOT_DIR,TRAIN_SET,batch_size)
    
    device = torch.device("cpu")
    
    stModel = model(spatialModel,temporalModel)



    learning_rate = conf["learning_rate"]

    my_loss_fn = lossFunction.HorusSpatioTemporalLoss()   # my custom spatiotemporal loss function


    optimizer = torch.optim.AdamW(stModel.parameters(),learning_rate)

    stModel.train()

    epochs_history = []
    
    min_loss = 1

    avg_loss = 1

    

    for k in range(epochs):
        batch_history = []

        if verbose:
            logger.info(f"SPATIOTEMPORAL TRAINING EPOCH: {k+1} of {epochs}")
        for batch,(imgs,labels) in enumerate(train):
            
            x_spatial,x_temporal = imgs
            y_spatial,y_temporal = labels

            x_spatial = x_spatial.to(device)                    #spatial image  student
            x_temporal = x_temporal.to(device)                  #temporal image student

            y_spatial = y_spatial.to(device)                    #spatial label  student
            y_temporal = y_temporal.to(device)                  #temporal label student

            
            x_temporal = torch.cat((x_spatial,x_temporal))      #create temporal student image stack
            y_temporal = torch.cat((y_spatial,y_temporal))      #create temporal student label stack
            
            optimizer.zero_grad()


            predict = stModel(x_spatial,x_temporal)


            loss = my_loss_fn(predict,y_spatial)

            
            loss.backward()

            optimizer.step()

            loss = loss.item()

            if batch % batch_saving == 0 and batch != 0:
                
                latest_batch = batch_history[-batch_saving:]

                avg_l = sum(latest_batch)/batch_saving            #average loss of batch length
                min_l = min(latest_batch)                         #min loss of batch length


                if verbose:
                    logger.info(f"Avg SpatioTemporal Loss Batch {batch}: {avg_l} ¦ Min Loss: {min_l}")

                if avg_l < avg_loss:
                    if verbose:
                        logger.info(f"Saving SpatioTemporal Model...")
                    torch.save(stModel.state_dict(), f"{DIR}/models/{files['ModelSpatioTemporal']}")
                    avg_loss = avg_l
                
            
            if loss < min_loss:         #if the spatial loss is less then the minumum spatial loss
                if verbose:
                    logger.info(f"Saving SpatioTemporal Model for min Loss: {loss}...")
                torch.save(stModel.state_dict(), f"{DIR}/models/{files['ModelSpatioTemporal']}")        #save the model
                min_loss = loss


        epochs_history.append(batch_history)

        batch_history = []

    history_dict = {}

    for i in range(len(batch_history)):
        history_dict[i] = batch_history[i]
    
    with open(f"{DIR}/json/{files['LossHistorySpatioTemporal']}","w") as jswr:     #save spatial loss history in json file
        json.dump(history_dict,jswr)
    
    return
    

