#!/usr/bin/env python
# coding: utf-8

from data_loader import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import shutil
import pickle



def plotting_func_1( x, y, xlabel="Epochs", ylabel="Loss", ylegend="Validation Accuracy",title="Accuracy vs Epochs curve",):
    """
    Plotting function for one graph in figure

    """
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend((ylegend), loc='best')
    plt.show()
    

def plotting_func_2( x, y1,y2, xlabel="Epochs", ylabel="Loss", y1legend="Validation Accuracy",y2legend ="Training Accuracy",title="Accuracy vs Epochs curve",):
    """
    Plotting function for two graphs in figure
    """
    plt.figure()
    plot1, = plt.plot(x, y1)
    plot2, = plt.plot(x, y2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    
    plt.grid(True)
    plt.legend((y1legend,y2legend), loc='best')
    plt.show()




class Vanilla_RNN(nn.Module):
    """ A basic RNN model 
    """
    
    def __init__(self, in_dim, out_dim, hid_dim, batch_size, no_layers =1):
        super(Vanilla_RNN, self).__init__()
        #specify the input dimensions
        self.in_dim = in_dim
        #specify the output dimensions
        self.out_dim = out_dim
        #specify hidden layer dimensions
        self.hid_dim = hid_dim
        #specify the number of layers
        self.no_layers = no_layers  
        self.batch_size=batch_size  #previously commented out
        
        #initialise the LSTM
        self.model = nn.RNN(self.in_dim, self.hid_dim, self.no_layers,nonlinearity = 'tanh')
        self.outputs = nn.Linear(self.hid_dim, out_dim)

    def forward(self, batch,hidden=None):
        """Pass the batch of images through each layer of the network, applying 
        """
        
        rnn_out, hidden = self.model(batch, hidden)
        y_pred = self.outputs(rnn_out.view(rnn_out.shape[0]*rnn_out.shape[1],rnn_out.shape[-1]))

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
#         hidden[0].detach_() 
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
num_epochs =  100  #100           # Number of full passes through the dataset
batch_size = 1 #1         # Number of samples in each minibatch
learning_rate = 0.001  

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

model = Vanilla_RNN(in_dim=94, out_dim=94,hid_dim=100,batch_size=1,no_layers=1)
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
#         print("hidden shape",hidden.size())
#         hidden[0].detach_() 
#         hidden[1].detach_()
#         hidden[0].no_grad()
#         hidden[1].no_grad()
#         hidden[0].detach() 
#         hidden[1].detach()
        hidden.detach_()
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
    #test
    loss_train_list.append(np.mean(avg_minibatch_loss))
    avg_minibatch_loss= []
    #test end
    
    print('val_loss: %.3f, average accuracy of all classes: %.3f'%(loss_val.avg,(acc_val.avg).mean()))
    
    # remember best loss and save checkpoint
    is_best = loss_val.avg <= best_loss
    best_loss = min(loss_val.avg, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
    
print("Training complete after", epoch, "epochs")

results = { "acc_train_list": acc_train_list, "acc_val_list": acc_val_list, "loss_train_list":loss_train_list,"loss_val_list":loss_val_list}
pickle.dump( results, open( "own_results_base.p", "wb" ) )


