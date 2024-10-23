"""
Tester file including auxiliary classes
"""
from . import *

class ANSIColors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

class BoolModelTester:
    teacherModel:bool
    studentModel:bool
    def __init__(self,b1,b2) -> None:
        self.teacherModel = b1
        self.studentModel = b2

class SaliencyTester:
    controllUnit:bool
    modelFileTeacher:str
    modelFileStudent:str
    modelUsages:BoolModelTester
    predictionAccuracy:BoolModelTester
    trainSetExample:BoolModelTester
    trainSetExample:BoolModelTester
    validSetExample:BoolModelTester
    testSetExample:BoolModelTester
    plotDisplayer:BoolModelTester
    def __init__(self,conf):
        self.controllUnit = conf["controllUnit"]
        self.modelFileTeacher = conf["modelFileTeacher"]
        self.modelFileStudent = conf["modelFileStudent"]
        self.modelUsages = BoolModelTester(conf["modelUsages"]["teacherModel"],conf["modelUsages"]["studentModel"])
        self.predictionAccuracy = BoolModelTester(conf["predictionAccuracy"]["teacherModel"],conf["predictionAccuracy"]["studentModel"])
        self.trainSetExample = BoolModelTester(conf["trainSetExample"]["teacherModel"],conf["trainSetExample"]["studentModel"])
        self.validSetExample = BoolModelTester(conf["validSetExample"]["teacherModel"],conf["validSetExample"]["studentModel"])
        self.testSetExample = BoolModelTester(conf["testSetExample"]["teacherModel"],conf["testSetExample"]["studentModel"])
        self.plotDisplayer = BoolModelTester(conf["plotDisplayer"]["teacherModel"],conf["plotDisplayer"]["studentModel"])


class TestControll:
    saliencyDetection:SaliencyTester
    def __init__(self,conf) -> None:
        self.saliencyDetection = SaliencyTester(conf["saliencyDetection"])

        

def controllUnitSaliency():
    """
    Importing Errors
    """
    import saliencyDetection
    from saliencyDetection import lossFunction,modelClasses,saliency,training
    from saliencyDetection.dataLoader import dataLoader
    from saliencyDetection.utils import displayer,model
    return

def modelUsages(modelFile):
    from saliencyDetection.modelClasses import Horus,HorusModelTeacherSpatial

    model = Horus(HorusModelTeacherSpatial,model_file=modelFile)
    checkPoint = model.getCheckPoint()
    state_dict = model.getStateDict()
    checkpoint_state_dict = checkPoint.getStateDict()
    assert (state_dict == checkpoint_state_dict), "state_dict of the model class should be the same as the state_dict of the checkpoint class"
    checkpoint_epoch = checkPoint.getEpoch()
    assert (checkpoint_epoch >= 1), f"The checkpoint epoch should be at least 1. Got: {checkpoint_epoch}"
    checkpoint_loss = checkPoint.getLoss()
    assert (type(checkpoint_loss) == list), f"The saved loss should be a list. Got: {type(checkpoint_loss)}"
    checkpoint_optimizer = checkPoint.getOptimizer()
    assert (checkpoint_optimizer != None), f"The optimizer must exists!"
    return

def predictionAccuracy(modelFile:str,model,dataset):
    from saliencyDetection.modelClasses import Horus
    import random
    from torchvision.transforms import Resize
    import numpy as np

    model = Horus(model,model_file=modelFile).getModel()
    dataset = dataset("2020-TIP-Fu-MMNet","testSet")
    accuracy = 0
    for _ in range(10):
        img,lab = dataset.__getitem__(random.randint(0,len(dataset)))
        pred = model(img)
        res = Resize((img.size(1),img.size(2)))
        diff = np.abs(res(pred).detach().numpy() - lab.detach().numpy())
        accuracy += np.mean(diff)
    accuracy = 1-(accuracy/10)
    assert (accuracy>0.8), f"The accuracy is below 80% ({accuracy*100:.2f}%). Need more training!!"
    return accuracy

def runExample(modelFile:str,model,dataset,run_type:str,modelType:str):
    from saliencyDetection.modelClasses import Horus
    from saliencyDetection.dataLoader.dataLoader import newLoader
    from saliencyDetection.utils import displayer

    horus = Horus(model,model_file=modelFile)
    dataloader = newLoader(dataset,"2020-TIP-Fu-MMNet",run_type,batch_size=4)
    p = displayer.Printer(f"{run_type.lower()}_{modelType}_tester.png")
    for _,(imgs,labs) in enumerate(dataloader):
        preds = horus.predict(imgs)
        p.save(imgs,preds,labs)
        break
    return

def plotDisplayerLog(logFile:str,file_name:str,exp:int=0):
    from saliencyDetection.utils import displayer

    p = displayer.Printer(file_name)
    p.fromLogFile(logFile,exp=exp)

def plotDisplayer(loss,file_name:str,exp:int=0):
    from saliencyDetection.utils import displayer

    p = displayer.Printer(file_name)
    p.lossChart(loss,exp=exp)
    

def controllUnitCollider():
    import wrapper
    from wrapper import collisions
    return

def collisionTest():
    from wrapper import collisions
    borders, objects = collisions.collider("Human_00196/Human_00196_00005.png","Human_00196/Human_00196_00010.png","Human_00196/Human_00196_00020.png")
    assert (len(borders) == 3 and len(objects) == 3), f"Should return 3,3 \nInstead of: {len(borders)},{len(objects)}"
    return

def testerCommandControll(conf:any):
    """
    Command&Controll for Horus testing functions
    """
    print("Tesing from conf file:\n")
    testControll = TestControll(conf)

    print("Saliency Detection...")
    saliency = testControll.saliencyDetection
    try:
        print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency.controllUnit}",end="\r")
        if saliency.controllUnit:
            controllUnitSaliency()
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency.controllUnit}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency.controllUnit}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency.controllUnit}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)
            
    modelFileTeacher = saliency.modelFileTeacher
    modelFileStudent = saliency.modelFileStudent

    from saliencyDetection.utils.model import CheckPoint
    checkpointT = CheckPoint(modelFileTeacher)
    assert (checkpointT.exists()), "Cannot continue the UnitTests if the model file doesn't exist"
    checkpointS = CheckPoint(modelFileStudent)
    assert (checkpointS.exists()), "Cannot continue the UnitTests if the model file doesn't exist"
    
    print("modelUsages:")
    try:
        print(f"\tTeacher:\t{saliency.modelUsages.teacherModel}",end="\r")
        if saliency.modelUsages.teacherModel:
            modelUsages(modelFileTeacher)
            print(f"\tTeacher:\t{saliency.modelUsages.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tTeacher:\t{saliency.modelUsages.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.modelUsages.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)
    
    try:
        print(f"\tStudent:\t{saliency.modelUsages.studentModel}",end="\r")
        if saliency.modelUsages.studentModel:
            modelUsages(modelFileStudent)
            print(f"\tStudent:\t{saliency.modelUsages.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tStudent:\t{saliency.modelUsages.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.modelUsages.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    from saliencyDetection.modelClasses import HorusModelTeacherSpatial, HorusModelStudentSpatial
    from saliencyDetection.dataLoader.dataLoader import AVS1KDataSetTeacherSpatial, AVS1KDataSetStudentSpatialOnly
    print("predictionAccuracy:")
    try:
        print(f"\tTeacher:\t{saliency.predictionAccuracy.teacherModel}",end="\r")
        if saliency.predictionAccuracy.teacherModel:
            acc = predictionAccuracy(modelFileTeacher,HorusModelTeacherSpatial,AVS1KDataSetTeacherSpatial)
            print(f"\tTeacher:\t{saliency.predictionAccuracy.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}] ({acc*100:.2f}%)")
        else:
            print(f"\tTeacher:\t{saliency.predictionAccuracy.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.predictionAccuracy.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"\tStudent:\t{saliency.predictionAccuracy.studentModel}",end="\r")
        if saliency.predictionAccuracy.studentModel:
            acc = predictionAccuracy(modelFileStudent,HorusModelStudentSpatial,AVS1KDataSetStudentSpatialOnly)
            print(f"\tStudent:\t{saliency.predictionAccuracy.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}] ({acc*100:.2f}%)")
        else:
            print(f"\tStudent:\t{saliency.predictionAccuracy.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.predictionAccuracy.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    print("trainSetExample:")
    try:
        print(f"\tTeacher:\t{saliency.trainSetExample.teacherModel}",end="\r")
        if saliency.trainSetExample.teacherModel:
            runExample(modelFileTeacher,HorusModelTeacherSpatial,AVS1KDataSetTeacherSpatial,"trainSet","teacher")
            print(f"\tTeacher:\t{saliency.trainSetExample.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tTeacher:\t{saliency.trainSetExample.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.trainSetExample.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"\tStudent:\t{saliency.trainSetExample.studentModel}",end="\r")
        if saliency.trainSetExample.studentModel:
            runExample(modelFileStudent,HorusModelStudentSpatial,AVS1KDataSetStudentSpatialOnly,"trainSet","student")
            print(f"\tStudent:\t{saliency.trainSetExample.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tStudent:\t{saliency.trainSetExample.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.trainSetExample.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    print("validSetExample:")
    try:
        print(f"\tTeacher:\t{saliency.validSetExample.teacherModel}",end="\r")
        if saliency.validSetExample.teacherModel:
            runExample(modelFileTeacher,HorusModelTeacherSpatial,AVS1KDataSetTeacherSpatial,"validSet","teacher")
            print(f"\tTeacher:\t{saliency.validSetExample.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tTeacher:\t{saliency.validSetExample.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.validSetExample.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"\tStudent:\t{saliency.validSetExample.studentModel}",end="\r")
        if saliency.validSetExample.studentModel:
            runExample(modelFileStudent,HorusModelStudentSpatial,AVS1KDataSetStudentSpatialOnly,"validSet","student")
            print(f"\tStudent:\t{saliency.validSetExample.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tStudent:\t{saliency.validSetExample.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.validSetExample.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    print("testSetExample:")
    try:
        print(f"\tTeacher:\t{saliency.testSetExample.teacherModel}",end="\r")
        if saliency.testSetExample.teacherModel:
            runExample(modelFileTeacher,HorusModelTeacherSpatial,AVS1KDataSetTeacherSpatial,"testSet","teacher")
            print(f"\tTeacher:\t{saliency.testSetExample.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tTeacher:\t{saliency.testSetExample.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.testSetExample.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"\tStudent:\t{saliency.testSetExample.studentModel}",end="\r")
        if saliency.testSetExample.studentModel:
            runExample(modelFileStudent,HorusModelStudentSpatial,AVS1KDataSetStudentSpatialOnly,"testSet","student")
            print(f"\tStudent:\t{saliency.testSetExample.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tStudent:\t{saliency.testSetExample.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.testSetExample.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    print("plotDisplayer:")
    try:
        print(f"\tTeacher:\t{saliency.plotDisplayer.teacherModel}",end="\r")
        if saliency.plotDisplayer.teacherModel:
            plotDisplayerLog(TEACHER_SPATIAL_LOG_FILE,"loss_plot_teacher_tester.png")
            print(f"\tTeacher:\t{saliency.plotDisplayer.teacherModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tTeacher:\t{saliency.plotDisplayer.teacherModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tTeacher:\t{saliency.plotDisplayer.teacherModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"\tStudent:\t{saliency.plotDisplayer.studentModel}",end="\r")
        if saliency.plotDisplayer.studentModel:
            loss = checkpointS.load().getLoss()
            plotDisplayerLog(STUDENT_SPATIAL_LOG_FILE,"loss_plot_student_tester.png",3)
            #plotDisplayer(loss,"prova.png",3)
            print(f"\tStudent:\t{saliency.plotDisplayer.studentModel}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"\tStudent:\t{saliency.plotDisplayer.studentModel}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"\tStudent:\t{saliency.plotDisplayer.studentModel}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    print("\nCollision Detection...")
    collider = conf["collider"]
    try:
        print(f"controllUnit:{' '*(23-len('controllUnit'))}{collider['controllUnit']}",end="\r")
        if collider["controllUnit"]:
            controllUnitCollider()
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{collider['controllUnit']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{collider['controllUnit']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{collider['controllUnit']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)

    try:
        print(f"collisionTest:{' '*(23-len('collisionTest'))}{collider['collisionTest']}",end="\r")
        if collider["collisionTest"]:
            collisionTest()
            print(f"collisionTest:{' '*(23-len('collisionTest'))}{collider['collisionTest']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"collisionTest:{' '*(23-len('collisionTest'))}{collider['collisionTest']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"collisionTest:{' '*(23-len('collisionTest'))}{collider['collisionTest']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)


