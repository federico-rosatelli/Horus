from dataLoader import dataLoader
from evaluation import evaluator
from objectDetection import visDrone



# d = Displayer(item=146,nframe=27)
# d.show()
# # d = Displayer()
# # d.show()
# # rembg

# d = dataLoader.Displayer()
# d.show()

# e = evaluator.Eval()
# e.newEval("public/images/ski_new.png")


# pp = e.getNextObj()
# e.showImage()
# e.showGaussianDiff()
# pp = e.getNextObj()
# e.showImage()

# vp = visDrone.VisDroneModel(device="cpu")
# predict,boxes = vp.predictImage("public/images/ski.png")
# vp.showImage(predict)