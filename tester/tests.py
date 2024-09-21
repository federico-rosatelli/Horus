from matplotlib import pyplot as plt
import wrapper.collisions as collisions
from saliencyDetection.modelClasses import Horus,HorusModelTeacherSpatial, HorusModelStudentSpatial
from saliencyDetection.dataLoader.dataLoader import AVS1KDataSetTeacherSpatial
from torch.utils.data import DataLoader


def unitTestCollider():
    borders, objects = collisions.collider("Human_00196/Human_00196_00005.png","Human_00196/Human_00196_00010.png","Human_00196/Human_00196_00020.png")
    assert (len(borders) == 3 and len(objects) == 3), f"Should return 3,3 \nInstead of: {len(borders)},{len(objects)}"
    return



def horus():
    h = Horus(HorusModelTeacherSpatial,model_file="horus_model_teacher_spatial.pt",state_dict=True)
    d = DataLoader(AVS1KDataSetTeacherSpatial("2020-TIP-Fu-MMNet","trainSet"), shuffle=True,batch_size=4)
    for _,(imgs,labels) in enumerate(d):

        img = imgs
        label = labels
    
        pred = h.predict(img)
        
        show(img,pred,label,"test_predict_new_teacher.png")
        break




def show(img,pred,label,file_name)->None:
    
    
    fig, axarr = plt.subplots(img.size(0),3)
    for i in range(len(img)):
        im = img[i].detach().numpy().transpose(1,2,0)
        pre = pred[i].detach().cpu().numpy().transpose(1,2,0)
        labe = label[i].detach().numpy().transpose(1,2,0)

        axarr[i][0].imshow((im))
        axarr[i][0].set_title('Image')
        axarr[i][0].axis('off')

        axarr[i][1].imshow(pre*255)
        axarr[i][1].set_title('Predict')
        axarr[i][1].axis('off')

        axarr[i][2].imshow((labe))
        axarr[i][2].set_title('Label')
        axarr[i][2].axis('off')

    
    plt.tight_layout()
    plt.savefig(f"testss/img/{file_name}")
    plt.clf()
    plt.close("all")
    plt.close(fig)
    plt.ioff()
