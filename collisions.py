#from run import Displayer
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.patches as patches


class CollisionsWrapper:
    def __init__(self,heatmatp_file) -> None:
        self.heatmap_resize = self.heatmapExtact(heatmatp_file)
        self.cv2_heatmap = cv2.imread(heatmatp_file)
        self.bias_h = 0.2
        self.bias_w = 0.4
        self.boxes = []
        self.obj = None
        pass

    def heatmapExtact(self,heatmap_file):
        im = Image.open(heatmap_file)
        im = im.resize((256,256))
        img = pil_to_tensor(im)
        img = img.permute(2, 0, 1)
        return img

    def show(self,img=None)->None:
        img = img if img.all() != None else self.cv2_heatmap
        fig, axarr = plt.subplots(1,2)
        axarr[0].imshow(img)
        axarr[0].set_title('Image')
        axarr[0].axis('off')
        axarr[1].imshow(self.heatmap_resize.permute(2,0,1))
        axarr[1].set_title('Image')
        axarr[1].axis('off')
        for box in self.boxes:
            rect = patches.Rectangle((box[0][0], box[0][1]), box[1][0], box[1][1], linewidth=1, edgecolor='r', facecolor='none')
            axarr[1].add_patch(rect)

        plt.tight_layout()
        plt.show()
    
    def boxCollision(self,box):
        gray = cv2.cvtColor(self.cv2_heatmap, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h,w,_ = self.cv2_heatmap.shape
        blank_image = np.zeros((h,w,3), np.uint8)
        cv2.drawContours(blank_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        self.show(blank_image)
        self.boxes.append(box)



b = cv2.imread("Human_00120_00141.png")
h = cv2.imread("Human_00120_00141_Ground.png")

added_image = cv2.addWeighted(b,0.5,h,0.8,0)

cv2.imwrite('combined.png', added_image)

# box = ([30,40],[100,150])
# c = CollisionsWrapper("pngegg.png")
# c.boxCollision(box)
#c.show()