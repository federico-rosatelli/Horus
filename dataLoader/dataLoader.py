from glob import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

class AVS1KDataloader:

    def __init__(self,rootDir,runType) -> None:
        self.class_list = ["Building","Human","Vehicle","Others"]
        
        if runType.lower() == "test":
            subDir = "testSet"
        elif runType.lower() == "valid":
            subDir = "validSet"
        else:
            subDir = "trainSet"
        self.video_Frame_List = sorted(glob(f"{rootDir}/{subDir}/Frame/*"))
        self.video_Ground_List = sorted(glob(f"{rootDir}/{subDir}/Ground/*"))

    def __getitem__(self,i:int):
        videospath = self.video_Frame_List[i]
        video = []
        for imgpath in sorted(glob(videospath+"/*")):
            # img = cv2.imread(imgpath)
            # img = cv2.resize(img, dsize = (256, 256))
            # img = img / 255
            # img = torch.from_numpy(img.astype(np.float32)).clone()
            #img = img.permute(2, 0, 1)
            im = Image.open(imgpath)
            im = im.resize((256,256))
            img = pil_to_tensor(im)
            img = img.permute(2, 0, 1)
            video.append(img)
        
        labelspaths = self.video_Ground_List[i]
        label = []
        for labelspath in sorted(glob(labelspaths+"/*")):
            im = Image.open(labelspath)
            im = im.resize((256,256))
            img = pil_to_tensor(im)
            img = img.permute(2, 0, 1)
            label.append(img)

        return video,label