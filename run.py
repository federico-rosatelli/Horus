from glob import glob
import random
from matplotlib import pyplot as plt
from dataLoader.dataLoader import AVS1KDataloader
from evaluation import evaluator




class Displayer:
    def __init__(self,loader:str="2020-TIP-Fu-MMNet",type:str="trainSet",item:int=-1,nframe:int=-1) -> None:
        self.loader = loader
        self.type = type
        length = len(glob(f"{loader}/{type}/Frame/*"))
        self.item = item if item >= 0 else random.randint(0,length)
        self.item = self.item if self.item < length else length - 1
        self.dataloader = AVS1KDataloader(loader,type)
        self.image,self.label = self.dataloader.__getitem__(self.item)
        self.nframe = nframe if nframe >= 0 else random.randint(0,len(self.image))
        self.nframe = self.nframe if self.nframe < len(self.image) else len(self.image)-1


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



# d = Displayer()
# d.show()
# rembg

e = evaluator.Eval(0.5,0.7,0.1,0.1)
e.newEval("public/images/boat.png")


pp = e.getNextObj()


e.showGaussianDiff()
e.showImage()