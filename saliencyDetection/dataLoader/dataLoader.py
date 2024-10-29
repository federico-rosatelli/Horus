"""
Custom Datasets for Horus Neural Network.

Each class is based on AVS1K Dataset.

This module implements :class:`DataLoader` from :module:`torch` module
"""
from glob import glob
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader


class AVS1KDataSetTeacherSpatial:
    size:tuple[int,int] = (980,460)
    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]

        #open spatial Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgSpatialGround = Image.open(spatialGroundPath)
        
        #resize spatial Frame and Ground
        imgSpatialFrame = imgSpatialFrame.resize(self.size)
        imgSpatialGround = imgSpatialGround.resize(self.size)

        #from PIL to Tensor spatial Frame and Ground
        imgSpatialFrame = pil_to_tensor(imgSpatialFrame)/255
        imgSpatialGround = pil_to_tensor(imgSpatialGround)/255

        return imgSpatialFrame, imgSpatialGround

class AVS1KDataSetTeacherTemporal:
    size:tuple[int,int] = (980,460)
    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]
        if i == len(self.all_Video_Frame)-1:
            temporalFramePath = self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i]
        else:
            temporalFramePath = self.all_Video_Frame[i+1] if (
                                self.all_Video_Frame[i+1].split("/")[-2] == spatialFramePath.split("/")[-2]
                                ) else self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i+1] if (
                                self.all_Video_Ground[i+1].split("/")[-2] == spatialGroundPath.split("/")[-2]
                                ) else self.all_Video_Ground[i]

        #open temporal Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgTemporalFrame = Image.open(temporalFramePath)
        
        imgSpatialGround = Image.open(spatialGroundPath)
        imgTemporalGround = Image.open(temporalGroundPath)

        #resize temporal Frame and Ground
        imgSpatialFrame = imgSpatialFrame.resize(self.size)
        imgTemporalFrame = imgTemporalFrame.resize(self.size)
        
        imgSpatialGround = imgSpatialGround.resize(self.size)
        imgTemporalGround = imgTemporalGround.resize(self.size)

        #from PIL to Tensor temporal Frame and Ground
        
        imgSpatialFrame = pil_to_tensor(imgSpatialFrame)/255
        imgTemporalFrame = pil_to_tensor(imgTemporalFrame)/255

        imgSpatialGround = pil_to_tensor(imgSpatialGround)/255
        imgTemporalGround = pil_to_tensor(imgTemporalGround)/255

        #channel dimension of temporal frame*2
        imgTemporalFrame = torch.cat((imgSpatialFrame,imgTemporalFrame),dim=0)
        imgTemporalGround = (imgSpatialGround*0.5)+(imgTemporalGround*0.5)

        return imgTemporalFrame, imgTemporalGround

class AVS1KDataSetStudentSpatialOnly:
    sizeS:tuple[int,int] = (30,14)

    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]

        #open spatial Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgSpatialGround = Image.open(spatialGroundPath)

        #resize spatial Frame and Ground for Teacher & Student

        imgSpatialFrameStudent = imgSpatialFrame.resize(self.sizeS)
        imgSpatialGroundStudent = imgSpatialGround.resize(self.sizeS)

        #from PIL to Tensor spatial Frame and Ground for Teacher & Student

        imgSpatialFrameStudent = pil_to_tensor(imgSpatialFrameStudent)/255
        imgSpatialGroundStudent = pil_to_tensor(imgSpatialGroundStudent)/255

        

        return imgSpatialFrameStudent,imgSpatialGroundStudent

class AVS1KDataSetStudentSpatial:
    sizeT:tuple[int,int] = (980,460)
    sizeS:tuple[int,int] = (30,14)

    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]

        #open spatial Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgSpatialGround = Image.open(spatialGroundPath)

        #resize spatial Frame and Ground for Teacher & Student
        imgSpatialFrameTeacher = imgSpatialFrame.resize(self.sizeT)
        imgSpatialGroundTeacher = imgSpatialGround.resize(self.sizeT)

        imgSpatialFrameStudent = imgSpatialFrame.resize(self.sizeS)
        imgSpatialGroundStudent = imgSpatialGround.resize(self.sizeS)

        #from PIL to Tensor spatial Frame and Ground for Teacher & Student
        imgSpatialFrameTeacher = pil_to_tensor(imgSpatialFrameTeacher)/255
        imgSpatialGroundTeacher = pil_to_tensor(imgSpatialGroundTeacher)/255

        imgSpatialFrameStudent = pil_to_tensor(imgSpatialFrameStudent)/255
        imgSpatialGroundStudent = pil_to_tensor(imgSpatialGroundStudent)/255

        

        return ((imgSpatialFrameTeacher, imgSpatialFrameStudent),
                (imgSpatialGroundTeacher,imgSpatialGroundStudent))



class AVS1KDataSetStudentTemporal:
    sizeT:tuple[int,int] = (980,460)
    sizeS:tuple[int,int] = (30,14)

    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]
        if i == len(self.all_Video_Frame)-1:
            temporalFramePath = self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i]
        else:
            temporalFramePath = self.all_Video_Frame[i+1] if (
                                self.all_Video_Frame[i+1].split("/")[-2] == spatialFramePath.split("/")[-2]
                                ) else self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i+1] if (
                                self.all_Video_Ground[i+1].split("/")[-2] == spatialGroundPath.split("/")[-2]
                                ) else self.all_Video_Ground[i]

        #open spatial/temporal Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgTemporalFrame = Image.open(temporalFramePath)
        
        imgSpatialGround = Image.open(spatialGroundPath)
        imgTemporalGround = Image.open(temporalGroundPath)

        #resize spatial/temporal Frame and Ground for Teacher & Student
        imgSpatialFrameTeacher = imgSpatialFrame.resize(self.sizeT)
        imgTemporalFrameTeacher = imgTemporalFrame.resize(self.sizeT)
        
        imgSpatialGroundTeacher = imgSpatialGround.resize(self.sizeT)
        imgTemporalGroundTeacher = imgTemporalGround.resize(self.sizeT)

        imgSpatialFrameStudent = imgSpatialFrame.resize(self.sizeS)
        imgTemporalFrameStudent = imgTemporalFrame.resize(self.sizeS)
        
        imgSpatialGroundStudent = imgSpatialGround.resize(self.sizeS)
        imgTemporalGroundStudent = imgTemporalGround.resize(self.sizeS)

        #from PIL to Tensor spatial/temporal Frame and Ground for Teacher & Student
        imgSpatialFrameTeacher = pil_to_tensor(imgSpatialFrameTeacher)/255
        imgTemporalFrameTeacher = pil_to_tensor(imgTemporalFrameTeacher)/255
        
        imgSpatialGroundTeacher = pil_to_tensor(imgSpatialGroundTeacher)/255
        imgTemporalGroundTeacher = pil_to_tensor(imgTemporalGroundTeacher)/255

        imgSpatialFrameStudent = pil_to_tensor(imgSpatialFrameStudent)/255
        imgTemporalFrameStudent = pil_to_tensor(imgTemporalFrameStudent)/255
        
        imgSpatialGroundStudent = pil_to_tensor(imgSpatialGroundStudent)/255
        imgTemporalGroundStudent = pil_to_tensor(imgTemporalGroundStudent)/255

        imgTemporalFrameTeacher = torch.cat((imgSpatialFrameTeacher,imgTemporalFrameTeacher),dim=0)
        imgTemporalGroundTeacher = (imgSpatialGroundTeacher*0.5)+(imgTemporalGroundTeacher*0.5)

        imgTemporalFrameStudent = torch.cat((imgSpatialFrameStudent,imgTemporalFrameStudent),dim=0)
        imgTemporalGroundStudent = (imgSpatialGroundStudent*0.5)+(imgTemporalGroundStudent*0.5)

        

        return ((imgTemporalFrameTeacher,imgTemporalFrameStudent),
                (imgTemporalGroundTeacher,imgTemporalGroundStudent))
    


class AVS1KDataSetStudentSpatioTemporal:
    sizeS:tuple[int,int] = (30,14)

    def __init__(self,rootDir,subDir) -> None:
        video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

        self.all_Video_Frame = []
        for video in video_Frame_List:
            self.all_Video_Frame += sorted(glob(f"{video}/*"))
        
        self.all_Video_Ground = []
        for ground in video_Ground_List:
            self.all_Video_Ground += sorted(glob(f"{ground}/*"))

    def __len__(self):
        return len(self.all_Video_Frame)
    
    def __getitem__(self,i):
        spatialFramePath = self.all_Video_Frame[i]
        spatialGroundPath = self.all_Video_Ground[i]
        if i == len(self.all_Video_Frame)-1:
            temporalFramePath = self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i]
        else:
            temporalFramePath = self.all_Video_Frame[i+1] if (
                                self.all_Video_Frame[i+1].split("/")[-2] == spatialFramePath.split("/")[-2]
                                ) else self.all_Video_Frame[i]
            temporalGroundPath = self.all_Video_Ground[i+1] if (
                                self.all_Video_Ground[i+1].split("/")[-2] == spatialGroundPath.split("/")[-2]
                                ) else self.all_Video_Ground[i]

        #open spatial/temporal Frame and Ground
        imgSpatialFrame = Image.open(spatialFramePath)
        imgTemporalFrame = Image.open(temporalFramePath)
        
        imgSpatialGround = Image.open(spatialGroundPath)
        imgTemporalGround = Image.open(temporalGroundPath)

        imgSpatialFrameStudent = imgSpatialFrame.resize(self.sizeS)
        imgTemporalFrameStudent = imgTemporalFrame.resize(self.sizeS)
        
        imgSpatialGroundStudent = imgSpatialGround.resize(self.sizeS)
        imgTemporalGroundStudent = imgTemporalGround.resize(self.sizeS)

        #from PIL to Tensor spatial/temporal Frame and Ground for Student

        imgSpatialFrameStudent = pil_to_tensor(imgSpatialFrameStudent)/255
        imgTemporalFrameStudent = pil_to_tensor(imgTemporalFrameStudent)/255
        
        imgSpatialGroundStudent = pil_to_tensor(imgSpatialGroundStudent)/255
        imgTemporalGroundStudent = pil_to_tensor(imgTemporalGroundStudent)/255


        imgTemporalFrameStudent = torch.cat((imgSpatialFrameStudent,imgTemporalFrameStudent),dim=0)
        imgTemporalGroundStudent = (imgSpatialGroundStudent*0.5)+(imgTemporalGroundStudent*0.5)

        

        return ((imgSpatialFrameStudent,imgTemporalFrameStudent),
                (imgSpatialGroundStudent,imgTemporalGroundStudent))




def newLoader(datasetCLass:any,rootDir:str, runType:str, batch_size:int=64) -> DataLoader:
    if runType.lower() == "test":
        subDir = "testSet"
    elif runType.lower() == "valid":
        subDir = "validSet"
    else:
        subDir = "trainSet"
    return DataLoader(datasetCLass(rootDir,subDir), shuffle=True,batch_size=batch_size)
