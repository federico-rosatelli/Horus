from matplotlib import pyplot as plt
import wrapper.collisions as collisions
from saliencyDetection.saliency import Horus,HorusModelTeacher
from saliencyDetection.dataLoader.dataLoader import AVS1KDataSet,AVS1KDataSetTeacher
from torch.utils.data import DataLoader


def unitTestCollider():
    borders, objects = collisions.collider("Human_00196/Human_00196_00005.png","Human_00196/Human_00196_00010.png","Human_00196/Human_00196_00020.png")
    assert (len(borders) == 3 and len(objects) == 3), f"Should return 3,3 \nInstead of: {len(borders)},{len(objects)}"
    return



def horus():
    h = Horus(HorusModelTeacher,model_file="horus_model_teacher_spatial.pt",state_dict=True)
    d = DataLoader(AVS1KDataSetTeacher("2020-TIP-Fu-MMNet","trainSet"), shuffle=True,batch_size=16)
    for _,(imgs,labels) in enumerate(d):

        img,_ = imgs
        label,_ = labels
        
        pred = h.predict(img[0])
        
        show(img[0],pred,label[0],"test_predict.png")
        break


def show(img,pred,label,file_name)->None:
    
    
    fig, axarr = plt.subplots(1,3)
    
    img = img.detach().numpy().transpose(0,2,1)
    pred = pred.detach().cpu().numpy().transpose(0,2,1)
    label = label.detach().numpy().transpose(0,2,1)

    axarr[0].imshow((img))
    axarr[0].set_title('Image')
    axarr[0].axis('off')

    axarr[1].imshow((pred*255))
    axarr[1].set_title('Predict')
    axarr[1].axis('off')

    axarr[2].imshow((label))
    axarr[2].set_title('Label')
    axarr[2].axis('off')

    
    plt.tight_layout()
    plt.savefig(f"testss/img/{file_name}")
    plt.clf()
    plt.close("all")
    plt.close(fig)
    plt.ioff()