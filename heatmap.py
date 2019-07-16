from data_loader1 import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import shutil
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plt


model = LSTM(in_dim=94, out_dim=94,hid_dim=100,batch_size=16,no_layers=1)
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

##amazing function to visualise color palette
sns.choose_diverging_palette()



with open('generated.txt', 'w') as f:
    for item in result:
        f.write("%s" % item)


''.join(result)


print("\n<start>\nX:47\nT:Prion Doulkad Coottlendy's\nR:polka\nK:G\nDD DD | Ad fg | BB BA | fd ed/e/ | fe dB/A/ |1 G3 :|2 \nA^c cA | d^c ed | ed (3ege | d2 B2 |\nef af | fa fe | fg fe | f3 gd ||\nP:variations\n|: a2 | afdf gedB | defd c2BA | BABd e2ce |1 fdfd cege ||\n|: dedB gedB | AGFE DBdf |\ngddB Agfe | dGFE gC |\nEG B/d/e/G/ cB/c/ | cc/B/ A>B | A2 ef/c/ | fe g>a | e>f e>e | fe c2 |\nB3 A/B/ | AG EF | B/c/d B2 | ec AG | B2 AG | B2 cB/A/ |\nBA A>B/A/ | GF Bd |1 A2 A>B :|2 A2 AG/A/ |\nd/f/e/d/ B/d/e |1 dA AG ||\nP:Variations varie ::\nDF (3ddd | =c2df | f2ef | fa (3g ef | \ngedB D2DE |\neAAB GAGB | A2AA BAGF |1 GABc de dc :|\n<end>")

