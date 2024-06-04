from glob import glob
import math
from matplotlib import pyplot as plt
from PIL import Image
import random
from torchvision.transforms.functional import pil_to_tensor

class AVS1KDataloader:

    def __init__(self,rootDir,subDir) -> None:
        self.video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        self.video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))
        k = math.inf
        for dir in glob(f"{rootDir}/{subDir}/Frame/*"):
            frames = glob(f"{dir}/*")
            k = len(frames) if len(frames) < k else k
        self.batch_size = k
        self.n_video = 0
        self.nextVideo()

    def getitem123(self,i:int):
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
    
    def __iter__(self):
        imgs_resize = []
        labels_resize = []

        if len(self.images_path) < self.batch_size:
            #print("QUI", len(self.images_path))
            self.nextVideo()

        #for _ in range(self.batch_size):
        img_path = self.images_path[0]
        img = Image.open(img_path)
        img_resize = img.resize((256,256))
        img_resize = pil_to_tensor(img_resize)
        img_resize = img_resize.permute(2,0,1)
        imgs_resize.append(img_resize/255)
        self.images_path = self.images_path[1:]

        label_path = self.labels_path[0]
        label = Image.open(label_path)
        label_resize = label.resize((256,256))
        label_resize = pil_to_tensor(label_resize)
        label_resize = label_resize.permute(2,0,1)
        labels_resize.append(label_resize/255)
        self.labels_path = self.labels_path[1:]
        
        yield img_resize/255,label_resize/255,self.images_path[:self.batch_size]
    
    def __len__(self) -> int:
        return sum([len(glob(f"{dir}/*")) for dir in self.video_Frame_List])
    
    def nextVideo(self) -> None:
        self.n_video += 1
        videosPath = self.video_Frame_List[self.n_video]
        labelPath = self.video_Ground_List[self.n_video]
        self.images_path = sorted(glob(videosPath+"/*"))
        self.labels_path = sorted(glob(labelPath+"/*"))



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