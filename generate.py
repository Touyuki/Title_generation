import torch.optim as optim
import torch.nn as nn
from datasets import title_m
from models import Generator,weights_init
import os
import torch
from torch.utils.data import DataLoader,Dataset
from args import get_args

os.environ['CUDA_VISIBLE_DEVICES']='1' ##
ngpu=1
args=get_args()
dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5}

def str_to_one_hot(str):
    lens=len(str)
    aa=torch.zeros(lens,42)
    for i in range(0,lens):
      new_t_vecs = torch.zeros(7, 6)
      for j in range(0,7):
        new_t_vecs[j][dit[str[i][j]]]=1
        aa1=new_t_vecs.view(1,-1)
        aa[i]=aa1
    return aa

netG=Generator(args.ngpu)
netG.apply(weights_init)
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta, 0.999))

db=title_m(args.datasets)

criterion = nn.L1Loss()

img_list = []
G_losses = []
D_losses = []
iters = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Starting Training Loop...")
# For each epoch

train_loader=DataLoader(db,batch_size=args.batch_size,shuffle=True)

for epoch in range(args.num_epochs):
    print("epoch {}/{}".format(epoch,args.num_epochs))
    print("-"*10)
    c_pointer=0
    for data in train_loader:
        img,label=data
        # print(label)
        int_ps=str_to_one_hot(label)
        int_ps2=int_ps.view(len(label),-1,1,1)
        out=netG.forward(int_ps2)
        errG=criterion(out,img.float())*100
        errG.backward()
        optimizerG.step()
        c_pointer+=1
        if c_pointer%20==0:
          print('epoch:{},batch:{},loss:{:.4f}'.format(epoch,c_pointer,errG.data.item()))

torch.save(netG, args.save_model+'netG.pkl')
torch.save(netG.state_dict(),args.save_model+"params.pkl")
