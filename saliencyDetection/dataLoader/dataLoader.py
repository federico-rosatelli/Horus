from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import random
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader


class AVS1KDataSetTeacher:
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
        videoPath = self.all_Video_Frame[i]
        groundPath = self.all_Video_Ground[i]
        
        imgFrame = Image.open(videoPath)
        imgGround = Image.open(groundPath)

        imgFrame = imgFrame.resize((720,1280))
        imgGround = imgGround.resize((720,1280))

        imgFrame = pil_to_tensor(imgFrame)
        imgGround = pil_to_tensor(imgGround)

        imgFrame = imgFrame.permute(2, 0, 1)
        imgGround = imgGround.permute(2, 0, 1)

        return imgFrame/255,imgGround/255

class AVS1KDataSet:

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
        imgSpatialFrameTeacher = imgSpatialFrame.resize((720,1280))
        imgTemporalFrameTeacher = imgTemporalFrame.resize((720,1280))
        
        imgSpatialGroundTeacher = imgSpatialGround.resize((720,1280))
        imgTemporalGroundTeacher = imgTemporalGround.resize((720,1280))

        imgSpatialFrameStudent = imgSpatialFrame.resize((256,256))
        imgTemporalFrameStudent = imgTemporalFrame.resize((256,256))
        
        imgSpatialGroundStudent = imgSpatialGround.resize((256,256))
        imgTemporalGroundStudent = imgTemporalGround.resize((256,256))

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
        imgSpatialFrameTeacher = imgSpatialFrameTeacher.permute(2, 0, 1)
        imgTemporalFrameTeacher = imgTemporalFrameTeacher.permute(2, 0, 1)

        imgSpatialGroundTeacher = imgSpatialGroundTeacher.permute(2, 0, 1)
        imgTemporalGroundTeacher = imgTemporalGroundTeacher.permute(2, 0, 1)

        imgSpatialFrameStudent = imgSpatialFrameStudent.permute(2, 0, 1)
        imgTemporalFrameStudent = imgTemporalFrameStudent.permute(2, 0, 1)

        imgSpatialGroundStudent = imgSpatialGroundStudent.permute(2, 0, 1)
        imgTemporalGroundStudent = imgTemporalGroundStudent.permute(2, 0, 1)

        return (
                (imgSpatialFrameTeacher/255,imgTemporalFrameTeacher/255),
                (imgSpatialFrameStudent/255,imgTemporalFrameStudent/25)),(
                (imgSpatialGroundTeacher/255,imgTemporalGroundTeacher/255),
                (imgSpatialGroundStudent/255,imgTemporalGroundStudent/25))



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

# def newTeacherLoader(rootDir:str, runType:str) ->DataLoader:
#     if runType.lower() == "test":
#         subDir = "testSet"
#     elif runType.lower() == "valid":
#         subDir = "validSet"
#     else:
#         subDir = "trainSet"
#     return DataLoader(AVS1KDataSetTeacher(rootDir,subDir), shuffle=True)

def newLoader(datasetCLass:any,rootDir:str, runType:str, batch_size:int=64) -> DataLoader:
    if runType.lower() == "test":
        subDir = "testSet"
    elif runType.lower() == "valid":
        subDir = "validSet"
    else:
        subDir = "trainSet"
    return DataLoader(datasetCLass(rootDir,subDir), shuffle=True,batch_size=batch_size)




class Displayer:
    def __init__(self,loader:str="2020-TIP-Fu-MMNet",type:str="trainSet",item:int=-1,nframe:int=-1) -> None:
        self.loader = loader
        self.type = type
        length = len(glob(f"{loader}/{type}/Frame/*"))
        self.item = random.randint(0,length) if item < 0 else item if item < length else length - 1
        #self.item = self.item if self.item < length else length - 1
        self.dataloader = AVS1KDataSet(loader,type)
        self.image,self.label = self.dataloader.__getitem__(self.item)
        self.nframe = random.randint(0,len(self.image)) if nframe < 0 else nframe if nframe < len(self.image) else len(self.image)-1
        #self.nframe = self.nframe if self.nframe < len(self.image) else len(self.image)-1


    def show(self)->None:
        fig, axarr = plt.subplots(1,2)
        axarr[0].imshow(self.image[self.nframe].permute(2,0,1))
        axarr[0].set_title('Image')
        axarr[0].axis('off')

        axarr[1].imshow(self.label[self.nframe].permute(2,0,1))
        axarr[1].set_title('Label')
        axarr[1].axis('off')

        fig.suptitle(f'Image & Label of {self.type}/{self.item} on Frame:{self.nframe}', fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def print(self) -> None:
        print()