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
import os
from pathlib import Path



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
            
            training.trainHorusTeacher_checkpoint(newConf,type_run="temporal",verbose=verbose)

    pathStudentSpatial = Path(f'{DIR}/{MODEL_DIR}/{studentConf["files"]["ModelSpatial"]}')  #student model
    
    if not pathStudentSpatial.exists():     #skip if model exists
        # Spatial Student Training
        teacherModelS = modelClasses.Horus(modelClasses.HorusModelTeacherSpatial,teacherConf["files"]["ModelSpatial"],device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentSpatial,modelClasses.HorusModelStudentSpatial,teacherModelS,type_run="spatial",verbose=verbose)

    
    pathStudentTemporal = Path(f'{DIR}/{MODEL_DIR}/{studentConf["files"]["ModelTemporal"]}')  #student model
    
    if not pathStudentTemporal.exists():     #skip if model exists
        # Temporal Student Training
        teacherModelT = modelClasses.Horus(modelClasses.HorusModelTeacherTemporal,teacherConf["files"]["ModelTemporal"],device=conf["device"])
        training.trainHorusStudent(conf,dtL.AVS1KDataSetStudentTemporal,modelClasses.HorusModelStudentTemporal,teacherModelT,type_run="temporal",verbose=verbose)
        
    print("===")
    




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
