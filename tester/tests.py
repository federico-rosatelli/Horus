
class ANSIColors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

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

def predictionAccuracy(modelFile:str):
    from saliencyDetection.modelClasses import Horus,HorusModelTeacherSpatial
    from saliencyDetection.dataLoader.dataLoader import AVS1KDataSetTeacherSpatial
    import random
    import numpy as np

    model = Horus(HorusModelTeacherSpatial,model_file=modelFile).getModel()
    dataset = AVS1KDataSetTeacherSpatial("2020-TIP-Fu-MMNet","testSet")
    accuracy = 0
    for _ in range(10):
        img,lab = dataset.__getitem__(random.randint(0,len(dataset)))
        pred = model(img)
        diff = np.abs(pred.detach().numpy() - lab.detach().numpy())
        accuracy += np.mean(diff)
    accuracy = 1-(accuracy/10)
    assert (accuracy>0.8), f"The accuracy is below 80% ({accuracy*100:.2f}%). Need more training!!"
    return accuracy

def runExample(modelFile:str,run_type:str):
    from saliencyDetection.modelClasses import Horus,HorusModelTeacherSpatial
    from saliencyDetection.dataLoader.dataLoader import AVS1KDataSetTeacherSpatial,newLoader
    from saliencyDetection.utils import displayer

    horus = Horus(HorusModelTeacherSpatial,model_file=modelFile)
    dataloader = newLoader(AVS1KDataSetTeacherSpatial,"2020-TIP-Fu-MMNet",run_type,batch_size=4)
    p = displayer.Printer(f"{run_type.lower()}_tester.png")
    for _,(imgs,labs) in enumerate(dataloader):
        preds = horus.predict(imgs)
        p.save(imgs,preds,labs)
        break
    return

def plotDisplayer(logFile:str):
    from saliencyDetection.utils import displayer

    p = displayer.Printer("loss_plot_tester.png")
    p.fromLogFile(logFile)
    

def controllUnitCollider():
    import wrapper
    from wrapper import collisions
    return

def collisionTest():
    from wrapper import collisions
    borders, objects = collisions.collider("Human_00196/Human_00196_00005.png","Human_00196/Human_00196_00010.png","Human_00196/Human_00196_00020.png")
    assert (len(borders) == 3 and len(objects) == 3), f"Should return 3,3 \nInstead of: {len(borders)},{len(objects)}"
    return

def testerCommandControll(conf):
    """
    Command&Controll for Horus testing functions
    """
    print("Tesing from conf file:")
    saliency = conf["saliencyDetection"]
    try:
        print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency['controllUnit']}",end="\r")
        if saliency["controllUnit"]:
            controllUnitSaliency()
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency['controllUnit']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency['controllUnit']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"controllUnit:{' '*(23-len('controllUnit'))}{saliency['controllUnit']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)
            
    from saliencyDetection.utils.model import CheckPoint
    checkpoint = CheckPoint("horus_model_teacher_spatial1.pt",".")
    assert (checkpoint.exists()), "Cannot continue the UnitTests if the model file doesn't exist"

    try:
        print(f"modelUsages:{' '*(23-len('modelUsages'))}{saliency['modelUsages']}",end="\r")
        if saliency["modelUsages"]:
            modelUsages("../../horus_model_teacher_spatial1.pt")
            print(f"modelUsages:{' '*(23-len('modelUsages'))}{saliency['modelUsages']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"modelUsages:{' '*(23-len('modelUsages'))}{saliency['modelUsages']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"modelUsages:{' '*(23-len('modelUsages'))}{saliency['modelUsages']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"predictionAccuracy:{' '*(23-len('predictionAccuracy'))}{saliency['predictionAccuracy']}",end="\r")
        if saliency["predictionAccuracy"]:
            acc = predictionAccuracy("../../horus_model_teacher_spatial1.pt")
            print(f"predictionAccuracy:{' '*(23-len('predictionAccuracy'))}{saliency['predictionAccuracy']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}] ({acc*100:.2f}%)")
        else:
            print(f"predictionAccuracy:{' '*(23-len('predictionAccuracy'))}{saliency['predictionAccuracy']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except AssertionError as ae:
            print(f"predictionAccuracy:{' '*(23-len('predictionAccuracy'))}{saliency['predictionAccuracy']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",ae)

    try:
        print(f"trainSetExample:{' '*(23-len('trainSetExample'))}{saliency['trainSetExample']}",end="\r")
        if saliency["trainSetExample"]:
            runExample("../../horus_model_teacher_spatial1.pt","trainSet")
            print(f"trainSetExample:{' '*(23-len('trainSetExample'))}{saliency['trainSetExample']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"trainSetExample:{' '*(23-len('trainSetExample'))}{saliency['trainSetExample']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"trainSetExample:{' '*(23-len('trainSetExample'))}{saliency['trainSetExample']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)

    try:
        print(f"validSetExample:{' '*(23-len('validSetExample'))}{saliency['validSetExample']}",end="\r")
        if saliency["validSetExample"]:
            runExample("../../horus_model_teacher_spatial1.pt","validSet")
            print(f"validSetExample:{' '*(23-len('validSetExample'))}{saliency['validSetExample']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"validSetExample:{' '*(23-len('validSetExample'))}{saliency['validSetExample']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"validSetExample:{' '*(23-len('validSetExample'))}{saliency['validSetExample']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)

    try:
        print(f"testSetExample:{' '*(23-len('testSetExample'))}{saliency['testSetExample']}",end="\r")
        if saliency["testSetExample"]:
            runExample("../../horus_model_teacher_spatial1.pt","testSet")
            print(f"testSetExample:{' '*(23-len('testSetExample'))}{saliency['testSetExample']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"testSetExample:{' '*(23-len('testSetExample'))}{saliency['testSetExample']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"testSetExample:{' '*(23-len('testSetExample'))}{saliency['testSetExample']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)

    try:
        print(f"plotDisplayer:{' '*(23-len('plotDisplayer'))}{saliency['plotDisplayer']}",end="\r")
        if saliency["plotDisplayer"]:
            plotDisplayer("logs/horus.log")
            print(f"plotDisplayer:{' '*(23-len('plotDisplayer'))}{saliency['plotDisplayer']}",f"\t[{ANSIColors.GREEN}V{ANSIColors.RESET}]")
        else:
            print(f"plotDisplayer:{' '*(23-len('plotDisplayer'))}{saliency['plotDisplayer']}",f"\t[{ANSIColors.BLUE}-{ANSIColors.RESET}]")
    except Exception as e:
            print(f"plotDisplayer:{' '*(23-len('plotDisplayer'))}{saliency['plotDisplayer']}",f"\t[{ANSIColors.RED}X{ANSIColors.RESET}]",e)

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


