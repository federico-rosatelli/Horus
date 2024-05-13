#from run import Displayer
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.patches as patches


class CollisionsWrapper:
    def __init__(self,heatmatp_file) -> None:
        self.heatmap = self.heatmapExtact(heatmatp_file)
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

    def show(self)->None:
        fig, axarr = plt.subplots()
        axarr.imshow(self.heatmap.permute(2,0,1))
        axarr.set_title('Image')
        axarr.axis('off')
        for box in self.boxes:
            rect = patches.Rectangle((box[0][0], box[0][1]), box[1][0], box[1][1], linewidth=1, edgecolor='r', facecolor='none')
            axarr.add_patch(rect)

        plt.tight_layout()
        plt.show()
    
    def boxCollision(self,box):
        self.boxes.append(box)


 

box = ([30,40],[100,150])
c = CollisionsWrapper("pngegg.png")
c.boxCollision(box)
c.show()