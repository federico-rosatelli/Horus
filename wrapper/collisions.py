#from run import Displayer
import warnings
warnings.simplefilter("ignore", UserWarning)
from . import *
import cv2
from uuid import *
import numpy as np
from evaluation import borderEval,evaluator
from objectDetection import visDrone
import subprocess
import os



class Wrapper:
    HOME_DIR = "public/images/temp/"
    def __init__(self,bias_border:float=0.2,saveFile:bool=False)-> None:
        self.bias_border = bias_border
        self.saveFile = saveFile
        self.history:list[borderEval.Segment] = []
        self.objKnown:list[borderEval.Segment] = []
        self.VisDrone = visDrone.VisDroneModel()
        self.eval = evaluator.Eval()
    
    def wrap(self,heatmap_tensor,input_image,temp_img=f"temp_{str(uuid4())}.png"):
        temp_img = self.HOME_DIR+temp_img
        borders = borderEval.borderSegments(heatmap_tensor)
        assert len(borders) > 0, "The heatmap must be present"
        cv2.imwrite(temp_img,input_image)
        img,boxes = self.VisDrone.predictImage(input_image)
        if self.saveFile:
            cv2.imwrite(f"{self.HOME_DIR}{temp_img}_boxes.png",img)

        borders,known = self.boxDiff(borders,boxes)
        self.objKnown += known
        if len(borders) == 0 and known:
            return None,None
            #raise Exception(f"Objects already known: {','.join([str(border) for border in known])}")
        
        try:
            filename_rembg = self.callRem(file_name=temp_img)
            rembg_image = cv2.imread(filename_rembg)
            
            rembg_borders = self.eval.newEval(rembg_image)
            
            not_rembg_borders = self.remDiff(borders,rembg_borders)
            

            self.eval.delObj(not_rembg_borders)

            obj = self.eval.getNextObj()
            
            self.history.append(obj)

            self.delImgs(filename_rembg,temp_img)
            return borders,obj
            

        except Exception as e:
            print(e)
            # self.history += borders
            return borders,None
    
    def getHistory(self):
        return self.history


    
    def callRem(self,file_name:str) -> str:
        filename_rembg = str(uuid4())
        try:
            subprocess.run(["rembg","i",file_name,f"{self.HOME_DIR}temp_{filename_rembg}.png"],shell=False,capture_output=False,text=False)
        except Exception as e:
            raise Exception(e)
        return f"{self.HOME_DIR}temp_{filename_rembg}.png"

    def delImgs(self,*files_name:str):
        for file_name in files_name:
            os.remove(file_name)

    def boxDiff(self,borders:list[borderEval.Segment],boxes:list[tuple]):
        bordersCheck = borders.copy()
        known = []
        for b in range(len(borders)):
            border = borders[b]
            bwImage = border.bwImage()
            area = len(border)
            size = border.size()
            for box in boxes:
                blank_image = np.zeros((size[0],size[1],3), np.uint8)
                blank_image = cv2.rectangle(blank_image,(box[0],box[1]),(box[2],box[3]),(255,255,255),-1)
                
                boxArea = (box[2]-box[0])*(box[3]-box[1])
                difference = bwImage - blank_image
                if self.saveFile:
                    cv2.imwrite(f"{self.HOME_DIR}{str(border)}_{b+1}.png",difference)
                differenceArea = np.count_nonzero(difference)

                if differenceArea < area - (boxArea*self.bias_border) and differenceArea < area*0.8:
                    bordersCheck.pop(b)
                    known.append((str(b),box))
                    break
        return bordersCheck,known
    
    def remDiff(self,borders:list[borderEval.Segment],rembg_borders:list[borderEval.Segment]):
        rembg_bordersCheck = []
        
        for b in range(len(borders)):
            border = borders[b]
            bwImage = border.bwImage()
            area = len(border)
            
            for rm_b in rembg_borders:
                rm_b_bwImage = rm_b.bwImage()
                rm_b_area = len(rm_b)
                difference = bwImage - rm_b_bwImage
                if self.saveFile:
                    cv2.imwrite(f"{self.HOME_DIR}{str(border)}_{b+1}_rembg.png",difference)
                differenceArea = np.count_nonzero(difference)

                if differenceArea < area - (rm_b_area*self.bias_border):
                    rembg_bordersCheck.append(rm_b)

        return [item for item in rembg_borders if item not in rembg_bordersCheck]



def collider(*files) -> tuple[list[borderEval.Segment], list[borderEval.Segment]]:
    base_img = f"{ROOT_DIR}/{TRAIN_SET}/Frame/"
    base_heatmap = f"{ROOT_DIR}/{TRAIN_SET}/Ground/"
    w = Wrapper(saveFile=False)
    
    borders_list:list[borderEval.Segment] = []
    objs_list:list[borderEval.Segment] = []
    for opFile in files:
        img = cv2.imread(f"{base_img}{opFile}")
        heatmap = cv2.imread(f"{base_heatmap}{opFile}")
        borders,obj = w.wrap(heatmap,img)
        if borders:
            borders_list += borders

        if obj:
            objs_list.append(obj)
    #print([(g.get("points"),g.get("pointVect")) for g in w.getHistory()])
    return borders_list, objs_list
    
