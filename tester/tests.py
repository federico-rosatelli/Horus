from matplotlib import pyplot as plt
import wrapper.collisions as collisions
from saliencyDetection.saliency import Horus
from saliencyDetection.dataLoader.dataLoader import AVS1KDataSet
from torch.utils.data import DataLoader


def unitTestCollider():
    borders, objects = collisions.collider("Human_00196/Human_00196_00005.png","Human_00196/Human_00196_00010.png","Human_00196/Human_00196_00020.png")
    assert (len(borders) == 3 and len(objects) == 3), f"Should return 3,3 \nInstead of: {len(borders)},{len(objects)}"
    return



def horus():
    h = Horus(model_file="horus_model.pt",state_dict=True)
    d = DataLoader(AVS1KDataSet("2020-TIP-Fu-MMNet","trainSet"), shuffle=True,batch_size=8)
    for _,(imgs,_) in enumerate(d):
        print(len(imgs))
        _,img = imgs
        
        pred = h.predict(img[0])
        show(img[0][0],pred[0],"test_predict.png")
        break


def show(img,pred,file_name,size=(256,256,1))->None:
    
    
    fig, axarr = plt.subplots(1,1)
    # if size == (256,256,1):
        
    #     img = (img*255).detach().cpu().numpy().transpose(0, 2, 1, 3)
        
    #     grayscale_transform = transforms.Grayscale(num_output_channels=1)  # Specify 1 channel for grayscale
    #     img = grayscale_transform(torch.from_numpy(np.array(img)).float())
    #     img = torch.from_numpy(np.array(img)).float().squeeze(0)
    img = img.detach().numpy().transpose(1,0,2)
    
    # axarr[0].imshow((img*255))
    # axarr[0].set_title('Image')
    # axarr[0].axis('off')
    pred = pred.reshape(256, 256)
    axarr.imshow((pred*255))
    axarr.set_title('Predict')
    axarr.axis('off')

    
    plt.tight_layout()
    plt.savefig(f"testss/img/{file_name}")
    plt.clf()
    plt.close("all")
    plt.close(fig)
    plt.ioff()