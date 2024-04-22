from glob import glob
import random
import cv2
from matplotlib import pyplot as plt
from dataLoader.dataLoader import AVS1KDataloader
from evaluation import evaluator
from objectDetection import visDrone




class Displayer:
    def __init__(self,loader:str="2020-TIP-Fu-MMNet",type:str="trainSet",item:int=-1,nframe:int=-1) -> None:
        self.loader = loader
        self.type = type
        length = len(glob(f"{loader}/{type}/Frame/*"))
        self.item = random.randint(0,length) if item < 0 else item if item < length else length - 1
        #self.item = self.item if self.item < length else length - 1
        self.dataloader = AVS1KDataloader(loader,type)
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


# d = Displayer(item=146,nframe=27)
# d.show()
# # d = Displayer()
# # d.show()
# # rembg

e = evaluator.Eval()
e.newEval("public/images/ski_new.png")


pp = e.getNextObj()
e.showImage()
pp = e.getNextObj()
e.showImage()

vp = visDrone.VisDroneModel(device="cpu")
predict,boxes = vp.predictImage("public/images/ski.png")
vp.showImage(predict)