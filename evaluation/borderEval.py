import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2
import numpy as np
from uuid import *
import skimage.exposure
import scipy.ndimage as ndi




class Segment:
    def __init__(self,dictSegm)->None:
        self.dictSegm = dictSegm
        self.id = str(uuid4())
    
    def __str__(self):
        return self.id
    
    def __call__(self):
        return self.dictSegm
    
    
    def __len__(self):
        return self.dictSegm["area"]
    
    def size(self):
        return self.dictSegm["size"]
    
    def addData(self,name:str,obj:any) -> None:
        self.dictSegm[name] = obj
    
    def bwImage(self):
        return self.dictSegm["bwImage"]
    
    def contours(self):
        return self.dictSegm["contours"]
    
    def gaussianDiff(self):
        return self.dictSegm["gaussianDiff"]
    
    def dominantColor(self):
        return self.dictSegm["dominantColor"]
    
    def get(self,name):
        if name not in self.dictSegm:
            raise Exception(f"Key name {name} not in Segment object {self.id}")
        return self.dictSegm[name]
    
def borderSegments(img) -> list[Segment]:
    #img = cv2.imread(png_file)

    h,w,_ = img.shape


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minArea = ((h*w)/100)*0.8

    segments = []
    for con in contours:
        blank_image = np.zeros((h,w,3), np.uint8)
        cv2.drawContours(blank_image, [con], -1, (255, 255, 255), thickness=cv2.FILLED)
        area = np.count_nonzero(blank_image)

        if area > minArea:
            masked = cv2.bitwise_and(blank_image,img)
            contObj = {
                "contours" : con,
                "img":img,
                "area" : area,
                "size": (h,w),
                "totArea": h*w,
                "bwImage": blank_image,
                "gaussianDiff":gaussianBorder(blank_image),
                "symmetry":objectSymmetry(blank_image,con),
                "dominantColor":dominantColor(masked)
            }
            segments.append(Segment(contObj))

    return segments

def gaussianBorder(img):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=20, sigmaY=20, borderType = cv2.BORDER_DEFAULT)
    result = skimage.exposure.rescale_intensity(blur, in_range=(80,255), out_range=(0,255))


    #diff = result - img

    return result

def objectSymmetry(img,con): #TODO
    
    new_con = con.reshape(con.shape[0],2)
    img1 = img.mean(axis=-1).astype('int')
    cy, cx = ndi.center_of_mass(img1)
    distance_metric = lambda x: (x[0] - int(cx))**2 + (x[1] - int(cy))**2
    farthest = max(new_con,key=distance_metric)
    closest = min(new_con,key=distance_metric)
    

    #### Object symmetry TODO ###
    shape = img.shape
    mask = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    rect_vertices = [
        (0,0),
        (0,shape[1]),
        (shape[0],shape[1]),
        (shape[0],0)
    ]
    for i in range(len(rect_vertices)):
        x3, y3 = rect_vertices[i]
        x4, y4 = rect_vertices[(i + 1) % len(rect_vertices)]

    # image = cv2.circle(img, (int(cx),int(cy)), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(image, (farthest[0],farthest[1]), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(image, (closest[0],closest[1]), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.line(image, (int(cx),int(cy)), (farthest[0],farthest[1]), color=(0, 0, 255), thickness=1)
    # cv2.line(image, (int(cx),int(cy)), (closest[0],closest[1]), color=(0, 0, 255), thickness=1)
    # cv2.imshow("asd",image)
    # cv2.waitKey(0)
    
    ### END TODO ###
    
    return {
        "center": (cx,cy),
        "farthestP": farthest,
        "closestP":closest
    }

def dominantColor(img):
    # mask for black pixels
    imgs = np.float32(img.reshape(-1, 3))
    mask = np.any(imgs != 0, axis=1)
    pixels = imgs[mask] 

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    return (int(dominant[0]),int(dominant[1]),int(dominant[2]))