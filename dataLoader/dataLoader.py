from glob import glob
from matplotlib import pyplot as plt
from PIL import Image
import random
from torchvision.transforms.functional import pil_to_tensor

class AVS1KDataloader:

    def __init__(self,rootDir,subDir) -> None:
        self.video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        self.video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

    def __getitem__(self,i:int):
        videospath = self.video_Frame_List[i]
        video = []
        video_resize = []
        for imgpath in sorted(glob(videospath+"/*")):
            # img = cv2.imread(imgpath)
            # img = cv2.resize(img, dsize = (256, 256))
            # img = img / 255
            # img = torch.from_numpy(img.astype(np.float32)).clone()
            #img = img.permute(2, 0, 1)
            im = Image.open(imgpath)
            img = pil_to_tensor(im)
            img = img.permute(2, 0, 1)
            video.append(img)

            im_resize = im.resize((256,256))
            img_resize = pil_to_tensor(im_resize)
            img_resize = img_resize.permute(2, 0, 1)
            video_resize.append(img_resize)
        
        labelspaths = self.video_Ground_List[i]
        label = []
        label_resize = []
        for labelspath in sorted(glob(labelspaths+"/*")):
            im = Image.open(labelspath)
            img = pil_to_tensor(im)
            img = img.permute(2, 0, 1)
            label.append(img)

            im_resize = im.resize((256,256))
            img_resize = pil_to_tensor(im_resize)
            img_resize = img_resize.permute(2, 0, 1)
            label_resize.append(img_resize)

        return (video,video_resize),(label,label_resize)
    
    def __len__(self) -> int:
        return len(self.video_Frame_List)



def newLoader(rootDir:str, runType:str) -> AVS1KDataloader:
    if runType.lower() == "test":
        subDir = "testSet"
    elif runType.lower() == "valid":
        subDir = "validSet"
    else:
        subDir = "trainSet"
    return AVS1KDataloader(rootDir,subDir)




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