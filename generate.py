import torch.optim as optim
import torch.nn as nn
from datasets import title_m
from models import Generator,weights_init
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader,Dataset
from args import get_args

os.environ['CUDA_VISIBLE_DEVICES']='1' ##
ngpu=1
args=get_args()
# dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5}
dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}

def str_to_one_hot(str):
    lens=len(str)
    # aa=torch.zeros(lens,42)
    aa=torch.zeros(lens,189)
    for i in range(0,lens):
      # new_t_vecs = torch.zeros(7, 6) # generate strings from A to E
      new_t_vecs = torch.zeros(7, 27) # generate strings from A to Z
      for j in range(0,7):
        new_t_vecs[j][dit[str[i][j]]]=1
        aa1=new_t_vecs.view(1,-1)
        aa[i]=aa1
    return aa

netG=Generator(args.ngpu)
netG.apply(weights_init)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))

db=title_m(args.datasets)

criterion=args.loss # the loss function can be change in args.py
cr2=nn.L1Loss()
iters = 0
print("Starting Training Loop...")
# For each epoch

train_loader=DataLoader(db,batch_size=args.batch_size,shuffle=True)

for epoch in range(args.num_epochs):
    print("epoch {}/{}".format(epoch,args.num_epochs))
    print("-"*10)
    c_pointer=0
    for data in train_loader:
        img,label=data
        img =img.float()
        int_ps=str_to_one_hot(label)
        int_ps2=int_ps.view(len(label),-1,1,1)
        out=netG.forward(int_ps2)*255 # the output of the network is in [0,1]
        errG=criterion(out,img)
        tmp_errG=cr2(out,img)
        errG.backward()
        optimizerG.step()
        c_pointer+=1
        if c_pointer%30==0:
          print('epoch:{},batch:{},loss:{:.4f}'.format(epoch,c_pointer,errG.data.item()))
          print('epoch:{},batch:{},l1loss:{:.4f}'.format(epoch,c_pointer,tmp_errG.data.item())) # this l1 loss is only used to compared results between the training with the loss function of L1loss and MSEloss

# save the weight file
torch.save(netG, args.save_model+'netG26_anti_mse_0.pkl')
torch.save(netG.state_dict(),args.save_model+"params_anti_mse_0.pkl")
