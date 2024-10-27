# AUTHOR Federico Rosatelli
"""
`saliencyDetection.saliency` is a Horus module that implements and manages the entire 
`Horus` neural network, from training to its use.
"""
import logging
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False
from .dataLoader import dataLoader as dtL
from . import lossFunction,modelClasses,training
from .utils.model import CheckPoint
from . import *
import json
import os
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

    if not pathTeacherSpatial.exists():   #skip if model exists
        # Spatial Teacher Training
        training.trainHorusTeacher(conf,dtL.AVS1KDataSetTeacherSpatial,modelClasses.HorusModelTeacherSpatial,type_run="spatial",verbose=verbose)
        
    else:
        # Get CheckPoint Model and Start from that
        checkpoint = CheckPoint(teacherConf["files"]["ModelSpatial"]).load()
        if checkpoint.getEpoch() < teacherConf["epochs"]:
            newConf = checkpoint.exportConf(device=conf["device"],
                                            dataset_class=dtL.AVS1KDataSetTeacherSpatial,
                                            model_class=modelClasses.HorusModelTeacherSpatial, 
                                            epochs=teacherConf["epochs"],
                                            batch_size=teacherConf["batch_size"],
                                            learning_rate=teacherConf["learning_rate"],
                                            loss_function=lossFunction.getLossFunction(teacherConf["loss_class"]),
                                            batch_saving=teacherConf["batch_saving"])
            
            training.trainHorusTeacher_checkpoint(newConf,verbose=verbose)
            
    
    
    pathTeacherTemporal = Path(os.path.join(DIR,MODEL_DIR,teacherConf["files"]["ModelTemporal"]))
    
    if not pathTeacherTemporal.exists():    #skip if model exists
        # Temporal Teacher Training    
        training.trainHorusTeacher(conf,dtL.AVS1KDataSetTeacherTemporal,modelClasses.HorusModelTeacherTemporal,type_run="temporal",verbose=verbose)

    else:
        # Get CheckPoint Model and Start from that
        checkpoint = CheckPoint(teacherConf["files"]["ModelTemporal"]).load()
        if checkpoint.getEpoch() < teacherConf["epochs"]:
            newConf = checkpoint.exportConf(device=conf["device"],
                                            dataset_class=dtL.AVS1KDataSetTeacherTemporal,
                                            model_class=modelClasses.HorusModelTeacherTemporal, 
                                            epochs=teacherConf["epochs"],
                                            batch_size=teacherConf["batch_size"],
                                            learning_rate=teacherConf["learning_rate"],
                                            loss_function=lossFunction.getLossFunction(teacherConf["loss_class"]),
                                            batch_saving=teacherConf["batch_saving"])
            
            training.trainHorusTeacher_checkpoint(newConf,verbose=verbose)

    pathStudentSpatial = Path(f'{DIR}/{MODEL_DIR}/{studentConf["files"]["ModelSpatial"]}')  #student model
    if not pathStudentSpatial.exists():     #skip if model exists

        teacherModelS = modelClasses.Horus(modelClasses.HorusModelTeacherSpatial,teacherConf["files"]["ModelSpatial"],device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentSpatial,modelClasses.HorusModelStudentSpatial,teacherModelS,type_run="spatial",verbose=verbose)

        teacherModelT = modelClasses.Horus(modelClasses.HorusModelTeacherTemporal,teacherConf["files"]["ModelTemporal"],device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentTemporal,modelClasses.HorusModelStudentTemporal,teacherModelT,type_run="temporal",verbose=verbose)
        
    print("===")
    #TODO Spatial Temporal Model Training (???)




def runHorus(conf,video_file:str):
    import cv2
    from torchvision.transforms import Resize
    import numpy as np
    from PIL import Image
    from .modelClasses import Horus, HorusModelTeacherSpatial, HorusModelStudentSpatial
    from .utils import model

    video_file = os.path.join(TESTER_DIR,"videos",video_file)
    
    #horusTeacherModel = Horus(HorusModelTeacherSpatial,model_file=conf["teacher"]["training"]["files"]["ModelSpatial"],device=conf["device"])
    horusStudentModel = Horus(HorusModelStudentSpatial,model_file=conf["student"]["training"]["files"]["ModelSpatial"],device=conf["device"])
    
    

    vidcap = cv2.VideoCapture(video_file)
    width  = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(vidcap.get(cv2.CAP_PROP_FPS))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    resize = Resize((height,width))
    
    new_video_file = video_file.replace(".","_")
    
    # videoOut1 = cv2.VideoWriter(f'{new_video_file}_teacher_pred.avi', -1, 20.0, (height,width),False)
    videoOut2 = cv2.VideoWriter(f'{new_video_file}_student_pred.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    success, image = vidcap.read()

    k = 0

    while success:
        
        k += 1
        processed = k/length
        print("Processing ["+"="*int(processed*20)+" "*(20-int(processed*20))+"]","%.1f" % (processed*100)+"%",end="\r")
        
        #frame1 = model.from_cv2_to_tensor(image,size=(980,460))
        frame2 = model.from_cv2_to_tensor(image,size=(30,14))

        #pred1 = horusTeacherModel.predict(frame1)
        pred2 = horusStudentModel.predict(frame2)
        
        frame = (resize(pred2).permute(1,2,0).detach().numpy()*255).astype(np.uint8)
        frame = np.squeeze(frame, axis=-1)
        zero_mask = frame < 20
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        frame[zero_mask] = [0, 0, 0]

        new_frame = cv2.addWeighted(frame, 0.5, image, 1, 0)
        videoOut2.write(new_frame)

        success, image = vidcap.read()
    print()
    vidcap.release()
    videoOut2.release()
    return





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
    

