from . import *
from . import model
import os
from matplotlib import pyplot as plt

class Printer:
    def __init__(self,file_name:str,new_root_dir:str=None) -> None:
        self.root_dir = IMGS_DIR if not new_root_dir else new_root_dir
        self.file_name = os.path.join(self.root_dir, file_name)
    
    def __call__(self, new_file_name:str) -> NotImplementedError:
        self.file_name = os.path.join(self.root_dir, new_file_name)
    
    def save(self,img,pred,label):
        fig, axarr = plt.subplots(img.size(0),3)

        axarr[0][0].set_title('Image')
        axarr[0][1].set_title('Predict')
        axarr[0][2].set_title('Label')

        for i in range(len(img)):
            im = img[i].detach().numpy().transpose(1,2,0)
            pre = pred[i].detach().cpu().numpy().transpose(1,2,0)
            labe = label[i].detach().numpy().transpose(1,2,0)

            axarr[i][0].imshow((im))
            axarr[i][0].axis('off')

            axarr[i][1].imshow((pre*255))
            axarr[i][1].axis('off')

            axarr[i][2].imshow((labe))
            axarr[i][2].axis('off')

        
        plt.tight_layout()
        plt.savefig(self.file_name)
        plt.clf()
        plt.close("all")
        plt.close(fig)
        plt.ioff()
    
    def lossChart(self,loss:model.LossItems,exp:int=0) -> None:
        lmd = 10**exp
        yavg = loss.avg()
        ymin = loss.min()
        x = [i+1 for i in range(len(loss))]
        plt.plot(x,yavg,label="AVG LOSS")
        plt.plot(x,ymin,label="MIN LOSS")
        plt.xticks(x)
        for i in range(len(yavg)):
            plt.text(i+1,yavg[i],"%.3f" % yavg[i])
            #plt.text(i,max_losses[i],"%.3f" % max_losses[i])
            plt.text(i+1,ymin[i],"%.3f" % ymin[i])
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(f"Losses (10{get_exponential(exp)} scalar)")
        plt.savefig(self.file_name)
        plt.close("all")
        plt.ioff()
    
    def fromLogFile(self,log_file,exp:int=0) -> None:
        lmd = 10**exp
        rl = open(log_file).readlines()
        all_loss = []
        k = -1
        for i in range(len(rl)):
            if "EPOCH" in rl[i] and "CHECKPOINT" not in rl[i]:
                all_loss.append([])
                k += 1
            if "Avg Spatial Loss Batch" in rl[i]:
                loss = float(rl[i].split(" ")[14])
                all_loss[k].append(loss)
        
        avg_losses = [sum(ep)/len(ep)*lmd for ep in all_loss]
        max_losses = [max(ep)*lmd for ep in all_loss]
        min_losses = [min(ep)*lmd for ep in all_loss]
        
        x = [i+1 for i in range(len(avg_losses))]
        plt.plot(x,avg_losses,label="AVG LOSSES")
        #plt.plot([i for i in range(len(max_losses))],max_losses,label="MAX LOSSES")
        plt.plot(x,min_losses,label="MIN LOSSES")
        plt.xticks(x)
        for i in range(len(avg_losses)):
            plt.text(i+1,avg_losses[i],"%.3f" % avg_losses[i])
            #plt.text(i,max_losses[i],"%.3f" % max_losses[i])
            plt.text(i+1,min_losses[i],"%.3f" % min_losses[i])
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(f"Losses (10{get_exponential(exp)} scalar)")
        plt.savefig(self.file_name)
        plt.close("all")
        plt.ioff()
            
            

superscript_map = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹"}

def get_exponential(exp:int) -> str:
    chars = ""
    for c in str(exp):
        chars += superscript_map[c]
    return chars
