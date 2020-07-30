# assume yhat is [batch,10], and y is [batch,1] 
import torch
import numpy as np
from PIL import Image
from IPython.display import clear_output
import torch.optim as optim
from torch.nn.init import xavier_normal_ , uniform_
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# This is currently a CPU implemenation... 
# TODO make GPU compatible with pytorch api someday ..
def accuracy(yhat,y) :
    #print(yhat)
    #print(y)
    yhat = yhat.cpu()
    y = y.cpu()  
    yhat = [np.int(i.argmax()) for i in yhat]
    yhat = np.asarray(yhat)
    y = np.asarray(y)
    #print(yhat.size, yhat)
    #print(y.size,y)
    num_correct = np.sum(np.where(y == yhat,1,0))
    #print("Num Correct = {} out of {}".format(num_correct, y.size))
    return num_correct


def viz_network(epoch, curves,model,use_cuda) :
    clear_output(True)
    
    # Set the plot size here .. about 4units/row ()
    plt.figure(figsize=[20,8])
    # plt.figure().subplots_adjust(hspace=1, wspace=1)
    
    # First plot the loss curves and accuracy
    ax1 = plt.subplot(2,4,1)
    ax2 = plt.subplot(2,4,2)

    for key in curves:
        loss = [x[0] for x in curves[key]]
        acc =  [float(x[1])/float(x[2]) for x in curves[key]]
        #print("acc = {}".format(acc))
        #plt.scatter(range(len(curves[key])), curves[key], label=key, linewidths=1.0)
        #loss
        ax1.plot( loss, label=key)
        ax1.legend()
        ax1.set_title('Loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Loss')
        #acc
        ax2.plot( acc, label=key)
        ax2.legend()
        ax2.set_title('Accuracy')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('% correct')
    

    # plot just the first convolutional layer weight distribution
    for idx, (n,m) in enumerate(model.named_modules()):
        if(isinstance(m,torch.nn.modules.conv.Conv2d)) :
            #print(n,m, m.weight.size())
            conv_weights = m.weight.detach().cpu().numpy().flatten()            
            #print("epoch = {} num_cnv1w = {} cnv1w_avg = {}".format(epoch, len(conv_weights), np.mean(conv_weights)))
            plt.subplot(2,4,3)
            plt.title("Layer {} weights".format(n))
            plt.hist(conv_weights)
            
            # Plot the gradients if available
            plt.subplot(2,4,4)      
            plt.title("Layer {} gradients".format(n))
            try :
                conv_grads   =  m.weight.grad.detach().cpu().numpy().flatten()
                print("epoch: {} / Conv Layer1 gradient avg :{}".format(epoch, np.mean(conv_grads)))
                plt.hist(conv_grads)
            except(AttributeError) :
                print("No gradients yet.  Please be patient")
            break;
            
    # plot just the first fully connected layer weight distribution
    for idx, (n,m) in enumerate(model.named_modules()):
        if(isinstance(m,torch.nn.modules.linear.Linear)) :
            # print(m, m.weight.size())
            fc_weights = m.weight.detach().cpu().numpy().flatten()
            #print("epoch = {} num_fc1w = {} fc1w_avg = {}".format(epoch, len(fc_weights), np.mean(fc_weights)))
            plt.subplot(2,4,5)
            plt.title("Layer {} weights".format(n))
            plt.hist(fc_weights)
            # Plot the gradients if available
            plt.subplot(2,4,6)            
            plt.title("Layer {} gradients".format(n))
            try :
                fc_grads   =  m.weight.grad.detach().cpu().numpy().flatten()
                print("epoch: {} / Fully Connected Grad avg : {}".format(epoch, np.mean(fc_grads)))
                plt.hist(fc_grads)
            except(AttributeError) :
                print("No gradients yet.  Please be patient")
            

            break;

    plt.show()
       
