

from data_loader import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import shutil
import pickle

def toAbc1(numLabel,dictionary):
    #take in onehot encoded data and transform it to abc

    dic_val_list=list(dictionary.values())
    dic_key_list=list(dictionary.keys()) 
    abc=dic_key_list[dic_val_list.index(numLabel)]
    #can be commented    
    return abc


model = Vanilla_RNN(in_dim=94, out_dim=94,hid_dim=100,batch_size=16,no_layers=1)
model = model.to(computing_device)
checkpoint = torch.load('model_best.pth.tar')
begin_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_loss']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])



dictionary=pickle.load(open("dictionary_new.pkl","rb"))
# from bidict import bidict 
# reverse_dict = bidict(dictionary)
# reverse_dict.inverse

def sample(net, size, prime, dictionary):
        
    if(use_cuda):
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    net = net.to(computing_device)
    net.eval() 
    
    h = None
    chars=list(prime)
    lis=[]
    for ch in chars:
        chNum=dictionary[ch]
        lis.append(chNum)
    
    lis=np.array(lis)
    softmaxLayer=torch.nn.Softmax()
    #pdb.set_trace()
    #a=torch.tensor(np.squeeze(np.eye(94)[lis.reshape(-1)]).reshape([len(prime), 1, 94])).to(computing_device).float()
    #pdb.set_trace()
    inputPrim=torch.tensor(fromNumLabelToOneHot(lis,dictionary).reshape([len(prime), 1, 94])).to(computing_device).float()
    with torch.no_grad():
        char, h = net(inputPrim, None ) 
         
        softmaxOutput=func.softmax(char[-1,:])
        p_p=softmaxOutput.cpu()
        p_p = np.array(p_p) 
        Val=np.random.choice(94, 1,p=p_p.reshape(94,))
        chars.append(toAbc1(Val,dictionary))
        newInput=fromNumLabelToOneHot(Val,dictionary)
        for j in range(size):     
            print(j)
            inputPrimNew=torch.tensor(np.zeros((100,1,94),dtype=int))
            #pdb.set_trace()
            inputPrimNew[:-1,:]=inputPrim[1:,:]
            inputPrimNew[-1,:]=torch.tensor(newInput)
            inputPrim=inputPrimNew.to(computing_device).float()
            char,h=net(inputPrim.view(len(prime),1,94),h)
            softmaxOutput=func.softmax(char[-1,:])
            p_p=softmaxOutput.cpu()
            p_p = np.array(p_p) 
            Val=np.random.choice(94, 1,p=p_p.reshape(94,))
            chars.append(toAbc1(Val,dictionary))
            newInput=fromNumLabelToOneHot(Val,dictionary)
            #pdb.set_trace()
    return chars



result=(sample(model, 2000, '<start>\nX:19\nT:Dusty Miller, The\nR:hop jig\nD:Tommy Keane: The Piper\'s Apron\nZ:id:hn-slipjig-19\nM:9/8', dictionary))


with open('generated.txt', 'w') as f:
    for item in result:
        f.write("%s" % item)


''.join(result)



view_results = pickle.load(open("own_results_base.p","rb"))

plotting_func_2(range(num_epochs),view_results['acc_val_list'],view_results['acc_train_list'],xlabel="Epochs", ylabel="Accuracy", y1legend="Validation Accuracy",y2legend ="Training Accuracy",title="Accuracy vs Epochs curve",)


plotting_func_2(range(num_epochs),view_results['loss_val_list'],view_results['loss_train_list'],xlabel="Epochs", ylabel="Loss", y1legend="Validation Loss",y2legend ="Training Loss",title="Loss vs Epochs curve",)


def test(loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()
    
    # switch to evaluate mode
    model.eval()  
    hidden=None
    for minibatch_count, (images, labels) in enumerate(loader, 0):
        images=images.permute(1,0,2)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        images=images.float()
       
        outputs,hidden = model(images,hidden)
#         hidden[0].detach_()   Need to figure out why
#         hidden[1].detach_()
        hidden.detach_()
        
        labels=labels.view(-1)
        loss = criterion(outputs, labels)
        
        softmaxOutput=func.softmax(outputs,dim=1)
        modelDecision=torch.argmax(softmaxOutput,dim=1)
        accuracy = calculate_acc(modelDecision,labels)
        losses.update(loss.item(), labels.shape[0])
        acc.update(accuracy, labels.shape[0])

    return acc,losses


acc_test,loss_test = test(test_loader, model, criterion)


print(f"Test Accuracy : {acc_test.avg:.4f}, Test Loss : {loss_test.avg:.4f}")






