"""
Custom Datasets for Horus Neural Network.

Each class is based on AVS1K Dataset.

This module implements :class:`DataLoader` from :module:`torch` module
"""
from glob import glob
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader


class AVS1KDataSetTeacher:
    size:tuple[int,int] = (1280,720)
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

        #resize spatial/temporal Frame and Ground
        imgSpatialFrame = imgSpatialFrame.resize(self.size)
        imgTemporalFrame = imgTemporalFrame.resize(self.size)
        
        imgSpatialGround = imgSpatialGround.resize(self.size)
        imgTemporalGround = imgTemporalGround.resize(self.size)

        #from PIL to Tensor spatial/temporal Frame and Ground
        imgSpatialFrame = pil_to_tensor(imgSpatialFrame)
        imgTemporalFrame = pil_to_tensor(imgTemporalFrame)
        
        imgSpatialGround = pil_to_tensor(imgSpatialGround)
        imgTemporalGround = pil_to_tensor(imgTemporalGround)

        #permute spatial/temporal Frame and Ground
        imgSpatialFrame = imgSpatialFrame.permute(1, 0, 2)
        imgTemporalFrame = imgTemporalFrame.permute(1, 0, 2)

        imgSpatialGround = imgSpatialGround.permute(1, 0, 2)
        imgTemporalGround = imgTemporalGround.permute(1, 0, 2)

        return (imgSpatialFrame/255,imgTemporalFrame/255),(imgSpatialGround/255,imgTemporalGround/255)

class AVS1KDataSet:
    sizeT:tuple[int,int] = (1280,720)
    sizeS:tuple[int,int] = (256,256)

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
        imgSpatialFrameTeacher = pil_to_tensor(imgSpatialFrameTeacher)
        imgTemporalFrameTeacher = pil_to_tensor(imgTemporalFrameTeacher)
        
        imgSpatialGroundTeacher = pil_to_tensor(imgSpatialGroundTeacher)
        imgTemporalGroundTeacher = pil_to_tensor(imgTemporalGroundTeacher)

        imgSpatialFrameStudent = pil_to_tensor(imgSpatialFrameStudent)
        imgTemporalFrameStudent = pil_to_tensor(imgTemporalFrameStudent)
        
        imgSpatialGroundStudent = pil_to_tensor(imgSpatialGroundStudent)
        imgTemporalGroundStudent = pil_to_tensor(imgTemporalGroundStudent)

        #permute spatial/temporal Frame and Ground for Teacher & Student
        imgSpatialFrameTeacher = imgSpatialFrameTeacher.permute(1, 0, 2)
        imgTemporalFrameTeacher = imgTemporalFrameTeacher.permute(1, 0, 2)

        imgSpatialGroundTeacher = imgSpatialGroundTeacher.permute(1, 0, 2)
        imgTemporalGroundTeacher = imgTemporalGroundTeacher.permute(1, 0, 2)

        imgSpatialFrameStudent = imgSpatialFrameStudent.permute(1, 0, 2)
        imgTemporalFrameStudent = imgTemporalFrameStudent.permute(1, 0, 2)

        imgSpatialGroundStudent = imgSpatialGroundStudent.permute(1, 0, 2)
        imgTemporalGroundStudent = imgTemporalGroundStudent.permute(1, 0, 2)

        return (
                (imgSpatialFrameTeacher/255,imgTemporalFrameTeacher/255),
                (imgSpatialFrameStudent/255,imgTemporalFrameStudent/255)),(
                (imgSpatialGroundTeacher/255,imgTemporalGroundTeacher/255),
                (imgSpatialGroundStudent/255,imgTemporalGroundStudent/255))



class AVS1KDataSetStudent:

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

        #resize spatial/temporal Frame and Ground
        imgSpatialFrame = imgSpatialFrame.resize((256,256))
        imgTemporalFrame = imgTemporalFrame.resize((256,256))
        
        imgSpatialGround = imgSpatialGround.resize((256,256))
        imgTemporalGround = imgTemporalGround.resize((256,256))

        #from PIL to Tensor spatial/temporal Frame and Ground
        imgSpatialFrame = pil_to_tensor(imgSpatialFrame)
        imgTemporalFrame = pil_to_tensor(imgTemporalFrame)
        
        imgSpatialGround = pil_to_tensor(imgSpatialGround)
        imgTemporalGround = pil_to_tensor(imgTemporalGround)

        #permute spatial/temporal Frame and Ground
        imgSpatialFrame = imgSpatialFrame.permute(2, 0, 1)
        imgTemporalFrame = imgTemporalFrame.permute(2, 0, 1)

        imgSpatialGround = imgSpatialGround.permute(2, 0, 1)
        imgTemporalGround = imgTemporalGround.permute(2, 0, 1)

        return (imgSpatialFrame/255,imgTemporalFrame/255),(imgSpatialGround/255,imgTemporalGround/25)


def newLoader(datasetCLass:any,rootDir:str, runType:str, batch_size:int=64) -> DataLoader:
    if runType.lower() == "test":
        subDir = "testSet"
    elif runType.lower() == "valid":
        subDir = "validSet"
    else:
        subDir = "trainSet"
    return DataLoader(datasetCLass(rootDir,subDir), shuffle=True,batch_size=batch_size)
