import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2
import numpy as np
from uuid import *
import skimage.exposure
import scipy.ndimage as ndi



def borderSegments(png_file:str) -> dict:
    img = cv2.imread(png_file)

    h,w,_ = img.shape


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minArea = ((h*w)/100)*2

    contObj = {}
    for con in contours:
        area = cv2.contourArea(con)
        if area > minArea:
            blank_image = np.zeros((h,w,3), np.uint8)

            cv2.drawContours(blank_image, [con], -1, (255, 255, 255), thickness=cv2.FILLED)
            
            contObj[str(uuid4())] = {
                "contours" : con,
                "img":img,
                "area" : area,
                "totArea": h*w,
                "bwImage": blank_image,
                "gaussianDiff":gaussianBorder(blank_image),
                "symmetry":objectSymmetry(blank_image,con),
                "dominantColor":dominantColor(img)
            }

    return contObj



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
    
    shape = img.shape
    mask = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    
    # image = cv2.circle(img, (int(cx),int(cy)), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(image, (farthest[0],farthest[1]), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.circle(image, (closest[0],closest[1]), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.line(image, (int(cx),int(cy)), (farthest[0],farthest[1]), color=(0, 0, 255), thickness=1)
    # cv2.line(image, (int(cx),int(cy)), (closest[0],closest[1]), color=(0, 0, 255), thickness=1)
    # cv2.imshow("asd",image)
    # cv2.waitKey(0)
    return {
        "center": (cx,cy),
        "farthestP": farthest,
        "closestP":closest
    }

def dominantColor(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    return (int(dominant[0]*255),int(dominant[1]*255),int(dominant[2]*255))

#borderSegments("../human.png")