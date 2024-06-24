#pip install ultralyticsplus ultralytics
import warnings
warnings.simplefilter("ignore", UserWarning)
from evaluation.printer import Printer
from ultralyticsplus import YOLO, render_result
import cv2
import numpy as np



class VisDroneModel:
    def __init__(self,model:str="mshamrai/yolov8n-visdrone", conf:float=0.25, iou:float=0.45, agnostic_nms:bool=False, max_det:int=1000,device:str="cuda") -> None:
        self.model = YOLO(model)
        self.model.overrides['conf'] = conf
        self.model.overrides['iou'] = iou
        self.model.overrides['agnostic_nms'] = agnostic_nms
        self.model.overrides['max_det'] = max_det

        self.model = self.model.cpu() #if device == "cpu" else self.model.cuda()

        self.predicted = []
        
    
    def predictImage(self,image:any) -> tuple[np.ndarray,list[list[int]]]:
        results = self.model.predict(image, show_labels=False, show_conf=False)
        boxes = []
        for box in results[0].boxes.numpy():
            r = box.xyxy[0].astype(int)
            boxes.append(r)
        img = results[0].plot(labels=False,conf=False)
        self.predicted.append((img,boxes))
        return img,boxes
    
    def showImage(self,image:np.ndarray) -> None:
        Printer({}).showImage(image)
        return
