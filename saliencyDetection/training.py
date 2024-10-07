# AUTHOR Federico Rosatelli
"""
Train Horus CNN Model
"""
import logging
logging.getLogger("matplotlib").propagate = False
logging.getLogger("PIL.PngImagePlugin").propagate = False
logging.getLogger("matplotlib.font_manager").propagate = False
from matplotlib import pyplot as plt
from saliencyDetection.dataLoader import dataLoader as dtL
import saliencyDetection.lossFunction as lossFunction
from saliencyDetection.utils.model import CheckPoint
from saliencyDetection.utils.displayer import Printer
from . import *
import json
import torch
import os


def trainHorusTeacher(conf:any,datasetCLass:any,model:any,type_run:str="spatial",verbose:str|None=None) -> None:
    """
    Training of Horus Teacher Model.

    Spatial & Temporal Knowledge implemented.

    Args:
        conf: a configuration dictionary from config path.
        datasetClass: Horus DataLoader class from :file:`dataLoader` module.
        model: Horus Model Class.
        type_run: string, `spatial`|`temporal`
        verbose: a string for :class:`logging` module to return a logger with the specified name.
    """
    device = conf["device"]
    conf = conf["teacher"]["training"]

    files = conf["files"]

    if type_run.lower() == "temporal":
        model_save = files["ModelTemporal"]
    else:
        model_save = files["ModelSpatial"]

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
    
    p = Printer(f"{type_run}_min_pred.png")
    check = CheckPoint(model_save)

    train = dtL.newLoader(datasetCLass,ROOT_DIR,TRAIN_SET,batch_size)

    device = torch.device(device)

    model = model()   
    model = model.to(device)
    
    learning_rate = conf["learning_rate"]

    lossClassName = conf["loss_class"]

    loss_fn = lossFunction.getLossFunction(lossClassName)()
    
    optimizer = torch.optim.AdamW(model.parameters(),learning_rate)
    
    epochs_history = [[] for _ in epochs]
    min_loss = float('inf')
    avg_loss = float('inf')

    model.train()

    for k in range(epochs):
        if verbose:
            logger.info(f"TEACHER TRAINING EPOCH {type_run.upper()}: {k+1} of {epochs}")
        #batch_history = []
        
        for batch,(imgs,labels) in enumerate(train):

            x = imgs.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            
            predict = model(x)

            loss = loss_fn(predict,y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            
            epochs_history[k].append(loss)
            
            p.save(x[:4],predict[:4],y[:4])
            if batch % batch_saving == 0 and batch != 0:


                latest_batch = epochs_history[k][-batch_saving:]
                avg_batch = sum(latest_batch)/batch_saving
                min_batch = min(latest_batch)
                max_batch = max(latest_batch)

                if verbose:
                    logger.info(f"Avg {type_run.lower().title()} Loss Batch {batch}: {avg_batch} ¦ Min Loss: {min_batch} ¦ Max Loss: {max_batch}")
                    
                if avg_batch < avg_loss:
                    if verbose:
                        logger.info(f"Saving {type_run.lower().title()} Model...")
                    check.save(k+1,model,optimizer,epochs_history)
                    #torch.save(model.state_dict(), f"{DIR}/models/{model_save}")
                    avg_loss = avg_batch
                    
            if loss < min_loss:
                if verbose:
                    logger.info(f"Saving {type_run.lower().title()} Model for min Loss: {loss}...")
                check.save(k+1,model,optimizer,epochs_history)
                #torch.save(model.state_dict(), f"{DIR}/models/{model_save}")
                min_loss = loss

        # epochs_history.append(batch_history)
        # batch_history = []

    return


def trainHorusStudent(conf:any,datasetClass:any,model:any,teacherModel:any,type_run:str="spatial",verbose:str|None=None) -> None:
    """
    Training of Horus Student Model.\n
    Spatial & Temporal Knowledge implemented.

    Args:
        conf: a configuration dictionary from config path.
        datasetClass: Horus DataLoader class from :module:`saliencyDetection.dataLoader.dataLoader` module.
        model: Horus Model Class.
        teacherModel: Horus Teacher Model
        type_run: string, `spatial`|`temporal`
        verbose: a string for :class:`logging` module to return a logger with the specified name.
    """

    device = conf["device"]
    conf = conf["student"]["training"]
    files = conf["files"]

    if type_run.lower() == "temporal":
        model_save = files["ModelTemporal"]
        json_save = files["LossHistoryTemporal"]
    else:
        model_save = files["ModelSpatial"]
        json_save = files["LossHistorySpatial"]
    
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
    
    device = torch.device(device)
    
    model = model()           # temporal model for student
    model = model.to(device)

    learning_rate = conf["learning_rate"]
    
    my_loss_fn = lossFunction.HorusLossFunction()   # my custom loss function

    optimizer = torch.optim.AdamW(model.parameters(),learning_rate)   #optimizer for model

    epochs_history = []
    min_loss = 1
    avg_loss = 1

    model.train()

    for k in range(epochs):
        batch_history:list[int] = []
        batch_history:list[int] = []

        if verbose:
            logger.info(f"STUDENT TRAINING EPOCH {type_run.upper()}: {k+1} of {epochs}")
        for batch,(imgs,labels) in enumerate(train):
            
            teacher_x,student_x = imgs
            teacher_y,student_y = labels

            teacher_x = teacher_x.to(device)            #image  teacher
            student_x = student_x.to(device)            #image  student

            teacher_y = teacher_y.to(device)            #label  teacher
            student_y = student_y.to(device)            #label  student

            optimizer.zero_grad()

            predictS = model(student_x)                                 #student prediction
            teacherPred = teacherModel.predict(teacher_x)               #teacher prediction

            #we're going to apply the teacher prediction on the custom loss function

            loss = my_loss_fn(predictS,teacherPred,teacher_y)    #our custom loss function

            loss.backward()

            optimizer.step()

            loss = loss.item()

            batch_history.append(loss)

            if batch % batch_saving == 0 and batch != 0:
                
                latest_batch = batch_history[-batch_saving:]

                avg_batch = sum(latest_batch)/batch_saving              #average loss of batch length
                min_batch = min(latest_batch)                           #min loss of batch length
                max_batch = max(latest_batch)                           #max loss of batch length

                if verbose:
                    logger.info(f"Avg {type_run.lower().title()} Loss Batch {batch}: {avg_batch} ¦ Min Loss: {min_batch} ¦ Max Loss: {max_batch}")

                if avg_batch < avg_loss:      #if the average loss is less then the minimum average loss
                    if verbose:
                        logger.info(f"Saving {type_run.lower().title()} Model...")
                    torch.save(model.state_dict(), f"{DIR}/models/{model_save}")    #save the model
                    avg_loss = avg_batch
                
            
            if loss < min_loss:         #if the loss is less then the minumum loss
                if verbose:
                    logger.info(f"Saving {type_run.lower().title()} Model for min Loss: {loss}...")
                torch.save(model.state_dict(), f"{DIR}/models/{model_save}")        #save the model
                min_loss = loss


        epochs_history.append(batch_history)

        batch_history = []
    
    history_dict = {}

    for i in range(len(batch_history)):
        history_dict[i] = batch_history[i]
    
    with open(f"{DIR}/json/{json_save}","w") as jswr:     #save spatial loss history in json file
        json.dump(history_dict,jswr)
    
    with open(f"{DIR}/json/{json_save}","w") as jswr:    #save temporal loss history in json file
        json.dump(history_dict,jswr)

    return model



def trainHorusTeacher_checkpoint(conf:any,type_run="spatial",verbose:str|None=None):

    device = conf["device"]
    datasetCLass = conf["dataset_class"]
    model_class = conf["model_class"]()
    state_dict = conf["state_dict"]
    start_epoch = conf["epoch"]-1
    model_save = conf["file"]
    optimizer = conf["optimizer"]
    epochs_history = conf["tot_loss"]
    epochs = conf["pochs"]
    batch_size = conf["batch_size"]
    # learning_rate = conf["learning_rate"]
    loss_fn = conf["loss_function"]()
    batch_saving = conf["batch_saving"]

    if verbose:
        logger = logging.getLogger(verbose)

    if len(epochs_history) < epochs:
        epochs_history = [[] for _ in epochs]
    epochs_history[start_epoch] = []

    min_loss = float('inf')#min([min(epo) for epo in epochs_history if epo])
    avg_loss = float('inf')

    train = dtL.newLoader(datasetCLass,ROOT_DIR,TRAIN_SET,batch_size)

    model = model_class.load_state_dict(state_dict)

    device = torch.device(device)

    model.train()

    p = Printer(f"{type_run}_min_pred.png")
    check = CheckPoint(model_save)

    for k in range(start_epoch,epochs):
        if verbose:
            logger.info(f"TEACHER TRAINING EPOCH {type_run.upper()}: {k+1} of {epochs}")
        #batch_history = []
        
        for batch,(imgs,labels) in enumerate(train):

            x = imgs.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            
            predict = model(x)

            loss = loss_fn(predict,y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            
            epochs_history[k].append(loss)
            
            p.save(x[:4],predict[:4],y[:4])
            if batch % batch_saving == 0 and batch != 0:


                latest_batch = epochs_history[k][-batch_saving:]
                avg_batch = sum(latest_batch)/batch_saving
                min_batch = min(latest_batch)
                max_batch = max(latest_batch)

                if verbose:
                    logger.info(f"Avg {type_run.lower().title()} Loss Batch {batch}: {avg_batch} ¦ Min Loss: {min_batch} ¦ Max Loss: {max_batch}")
                    
                if avg_batch < avg_loss:
                    if verbose:
                        logger.info(f"Saving {type_run.lower().title()} Model...")
                    check.save(k+1,model,optimizer,epochs_history)
                    avg_loss = avg_batch
                    
            if loss < min_loss:
                if verbose:
                    logger.info(f"Saving {type_run.lower().title()} Model for min Loss: {loss}...")
                check.save(k+1,model,optimizer,epochs_history)
                min_loss = loss

    return