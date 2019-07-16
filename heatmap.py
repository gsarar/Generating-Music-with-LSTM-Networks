#!/usr/bin/env python
# coding: utf-8

# In[2]:


from data_loader1 import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import shutil
import pickle


# In[41]:




class LSTM(nn.Module):
    """ A basic LSTM model. 
    """
    
    def __init__(self, in_dim, out_dim, hid_dim, batch_size, no_layers =1):
        super(LSTM, self).__init__()
        #specify the input dimensions
        self.in_dim = in_dim
        #specify the output dimensions
        self.out_dim = out_dim
        #specify hidden layer dimensions
        self.hid_dim = hid_dim
        #specify the number of layers
        self.no_layers = no_layers  
        #self.batch_size=batch_size
        
        #initialise the LSTM
        self.model = nn.LSTM(self.in_dim, self.hid_dim, self.no_layers)
        self.intermediate_out=0
        self.outputs = nn.Linear(self.hid_dim, out_dim)
        
    def get_intermediate_weights(self):
        return self.intermediate_out

    def forward(self, batch,hidden=None):
        """Pass the batch of images through each layer of the network, applying 
        """
        
        lstm_out, hidden = self.model(batch, hidden)
        y_pred = self.outputs(lstm_out.view(lstm_out.shape[0]*lstm_out.shape[1],lstm_out.shape[-1]))
        self.intermediate_out=lstm_out
        #pdb.set_trace()
        #The input is expected to contain raw, unnormalized scores for each class according to documentation
        #tag_scores = func.softmax(y_pred,dim=2)
        #return tag_scores,hidden
        return y_pred,hidden

    
def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()  
    hidden=None
    for minibatch_count, (images, labels) in enumerate(val_loader, 0):
        images=images.permute(1,0,2)
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        images=images.float()
       
        outputs,hidden = model(images,hidden)
        hidden[0].detach_() 
        hidden[1].detach_()
        
        labels=labels.view(-1)
        loss = criterion(outputs, labels)
        
        softmaxOutput=func.softmax(outputs,dim=1)
        modelDecision=torch.argmax(softmaxOutput,dim=1)
        accuracy = calculate_acc(modelDecision,labels)
        losses.update(loss.item(), labels.shape[0])
        acc.update(accuracy, labels.shape[0])

    return acc,losses
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_acc(modelDecision,labels):
    acc=torch.sum(labels==modelDecision).to(dtype=torch.float)/labels.shape[0]
    return acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
        
# Setup: initialize the hyperparameters/variables
num_epochs = 100         # Number of full passes through the dataset
batch_size = 1          # Number of samples in each minibatch
learning_rate = 0.001  
#use_cuda=0
use_cuda=torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size,shuffle=False, show_sample=False,extras=extras)

# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
model = LSTM(in_dim=94, out_dim=94,hid_dim=100,batch_size=16,no_layers=1)
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

#TODO: Define the loss criterion and instantiate the gradient descent optimizer
criterion = torch.nn.CrossEntropyLoss() #TODO - loss criteria are defined in the torch.nn package

#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Track the loss across training
total_loss = []
avg_minibatch_loss = []
best_loss = 100
acc_train_list=[]
acc_val_list=[]

loss_train_list=[]
loss_val_list=[]

hidden=None

# Begin training procedure
for epoch in range(num_epochs):
    model.train(True)
    N = 50
    N_minibatch_loss = 0.0    
    train_acc = AverageMeter()
    train_precision = AverageMeter()
    train_recall = AverageMeter()
    train_BCR = AverageMeter()
    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):

        images=images.permute(1,0,2)

        
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        images=images.float()
        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()
        

        # Perform the forward pass through the network and compute the loss
        outputs,hidden = model(images,hidden)
        hidden[0].detach_() 
        hidden[1].detach_()
#         outputs.shape => batchSize * sequenceSize *dictionarySize
#         labels.shape => batchSize * sequenceSize
        labels=labels.view(-1)
        loss = criterion(outputs,labels)
        
        # Automagically compute the gradients and backpropagate the loss through the network
        #loss.backward(retain_graph=True)
        loss.backward()
        # Update the weights
        optimizer.step()

        #Calculate accuracy
        #pdb.set_trace()
        softmaxOutput=func.softmax(outputs,dim=1)
        modelDecision=torch.argmax(softmaxOutput,dim=1)
        accuracy = calculate_acc(modelDecision,labels)
        train_acc.update(accuracy, labels.shape[0])
        

        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        N_minibatch_loss += float(loss)
        
        #TODO: Implement cross-validation
        del outputs
        del loss
        
        if minibatch_count % N == 0:    
            
            # Print the loss averaged over the last N mini-batches    
            N_minibatch_loss /= N
            print('Epoch %d, average minibatch %d loss: %.3f, average accuracy: %.3f' %
                (epoch + 1, minibatch_count, N_minibatch_loss,train_acc.avg.mean()))
            
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            N_minibatch_loss = 0.0

            
    print("Finished", epoch + 1, "epochs of training")
    #TODO: Implement cross-validation
    with torch.no_grad():
        acc_val,loss_val = validate(val_loader, model, criterion)
    
    acc_train_list.append(train_acc.avg.mean())
    acc_val_list.append((acc_val.avg).mean())
    
    loss_val_list.append(loss_val.avg)
    print('val_loss: %.3f, average accuracy of all classes: %.3f'%(loss_val.avg,(acc_val.avg).mean()))
    
    # remember best loss and save checkpoint
    is_best = loss_val.avg <= best_loss
    best_loss = min(loss_val.avg, best_loss)
    print("done")
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
        'intermediate':model.get_intermediate_weights()
    }, is_best)
    
print("Training complete after", epoch, "epochs")

results = { "acc_train_list": acc_train_list, "acc_val_list": acc_val_list, "loss_train_list":loss_train_list,"loss_val_list":loss_val_list}
pickle.dump( results, open( "own_results_base.p", "wb" ) )


# In[44]:


mod=torch.load("checkpoint_done.pth.tar")


# In[46]:


def toAbc1(numLabel,dictionary):
    #take in onehot encoded data and transform it to abc

    dic_val_list=list(dictionary.values())
    dic_key_list=list(dictionary.keys()) 
    abc=dic_key_list[dic_val_list.index(numLabel)]
    #can be commented    
    return abc


# In[47]:


from data_loader1 import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import shutil
import pickle
model = LSTM(in_dim=94, out_dim=94,hid_dim=100,batch_size=16,no_layers=1)
model = model.to(computing_device)
checkpoint = torch.load('model_best.pth.tar')
begin_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_loss']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


# In[48]:


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
    intermediate_master=[]
    with torch.no_grad():
        char, h = net(inputPrim, None )
        
        intermediate=net.get_intermediate_weights()
        intermediate_master.append(intermediate)
        #pdb.set_trace()
         
        softmaxOutput=func.softmax(char[-1,:])
        p_p=softmaxOutput.cpu()
        p_p = np.array(p_p) 
        Val=np.random.choice(94, 1,p=p_p.reshape(94,))
        chars.append(toAbc1(Val,dictionary))
        newInput=fromNumLabelToOneHot(Val,dictionary)
        for j in range(size):     
#             print(j)
            inputPrimNew=torch.tensor(np.zeros((100,1,94),dtype=int))
            #pdb.set_trace()
            inputPrimNew[:-1,:]=inputPrim[1:,:]
            inputPrimNew[-1,:]=torch.tensor(newInput)
            inputPrim=inputPrimNew.to(computing_device).float()
            char,h=net(inputPrim.view(len(prime),1,94),h)
#             print("hereee")
            intermediate=net.get_intermediate_weights()
            intermediate_master.append(intermediate)
            softmaxOutput=func.softmax(char[-1,:])
            p_p=softmaxOutput.cpu()
            p_p = np.array(p_p) 
            Val=np.random.choice(94, 1,p=p_p.reshape(94,))
            chars.append(toAbc1(Val,dictionary))
            newInput=fromNumLabelToOneHot(Val,dictionary)
            #pdb.set_trace()
    return chars,intermediate_master
        
result,intermediate_master=(sample(model, 2000, '<start>\nX:19\nT:Dusty Miller, The\nR:hop jig\nD:Tommy Keane: The Piper\'s Apron\nZ:id:hn-slipjig-19\nM:9/8', dictionary))


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plt

###heat map generation
##p is the neuron index
for neuron in range(0,99,1):
    out=[]
## i in the intermediate cell
    for i in intermediate_master:
        last_complete_sequence=i[-1]
        last_complete_sequence=last_complete_sequence[0][neuron].item()
        out.append(last_complete_sequence)
    out=np.array(out)
    out=out[1:]
    res=np.array(result)
    res=res[101:]
    res.shape
    res
    fig, ax = plt.subplots(figsize=(12,12))         # Sample figsize in inches
    sns.set()
    plt.title(f"neuron i={p}")
    #cmap = sns.cm.rocket_r
    cmap = sns.diverging_palette(257,11,s=99, sep=10,n=16, as_cmap=True)
    ax = sns.heatmap(out.reshape(50,40),annot=res.reshape(50,40),fmt="s",ax=ax,cmap=cmap)
    plt.savefig(f"hot_neuron_i={p}")
    plt.show()
    fig


# In[33]:

##amazing function to visualise color palette
sns.choose_diverging_palette()


# In[ ]:


with open('generated.txt', 'w') as f:
    for item in result:
        f.write("%s" % item)


# In[13]:


''.join(result)


# In[14]:


print("\n<start>\nX:47\nT:Prion Doulkad Coottlendy's\nR:polka\nK:G\nDD DD | Ad fg | BB BA | fd ed/e/ | fe dB/A/ |1 G3 :|2 \nA^c cA | d^c ed | ed (3ege | d2 B2 |\nef af | fa fe | fg fe | f3 gd ||\nP:variations\n|: a2 | afdf gedB | defd c2BA | BABd e2ce |1 fdfd cege ||\n|: dedB gedB | AGFE DBdf |\ngddB Agfe | dGFE gC |\nEG B/d/e/G/ cB/c/ | cc/B/ A>B | A2 ef/c/ | fe g>a | e>f e>e | fe c2 |\nB3 A/B/ | AG EF | B/c/d B2 | ec AG | B2 AG | B2 cB/A/ |\nBA A>B/A/ | GF Bd |1 A2 A>B :|2 A2 AG/A/ |\nd/f/e/d/ B/d/e |1 dA AG ||\nP:Variations varie ::\nDF (3ddd | =c2df | f2ef | fa (3g ef | \ngedB D2DE |\neAAB GAGB | A2AA BAGF |1 GABc de dc :|\n<end>")

