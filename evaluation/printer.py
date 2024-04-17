import math
import numpy as np
import cv2

class Printer:
    def __init__(self,obj:dict) -> None:
        self.contours = obj["contours"]
        self.img = obj["img"]
        self.bwImage = obj["bwImage"]
        self.gaussianDiff = obj["gaussianDiff"]
    
    def showBwImage(self) -> None:
        cv2.imshow("Black/White Image",self.bwImage)
        cv2.waitKey(0)
        return
    
    def showImgContours(self) -> None:
        img = self.img.copy()
        cv2.drawContours(img, self.contours, -1, (0,0,255), 3)
        cv2.imshow("Object Contours",img)
        cv2.waitKey(0)
        return
    
    def showGaussianDiff(self) -> None:
        diff = self.gaussianDiff - self.bwImage
        cv2.imshow("Gaussian Difference",diff)
        cv2.waitKey(0)
        return
