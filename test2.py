from models import Generator
import torch
import random
from glob import glob
from args import get_args
from datetime import datetime
from torchvision import transforms
import matplotlib.image as Image
import os
dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
dic={ 0 :"0",1: "A",2: "B",3: "C",4: "D",5: "E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L",13:"M",14:"N",15:"O",16:"P",17:"Q",18:"R",19:"S",20:"T",21:"U",22:"V",23:"W",24:"X",25:"Y",26:"Z"}
def str_to_one_hot(str):
    lens=len(str)
    aa=torch.zeros(lens,189)
    for i in range(0,lens):
      new_t_vecs = torch.zeros(7, 27)
      for j in range(0,7):
        new_t_vecs[j][dit[str[i][j]]]=1
        aa1=new_t_vecs.view(1,-1)
        aa[i]=aa1
    return aa

args=get_args()
ngpu=1

netG2 = torch.load(args.save_model+'netG26_l1_1.pkl')

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
images = glob('test_data/train/*.png')
i=0
os.mkdir(args.results2 +current_time+"--"+str(args.loss))
loss=args.loss
total_loss=0


## generate the same strings with the training data
for image in images:
    fl_nm=image[-11:-4]
    fl_nm=[fl_nm]
    one_hot=str_to_one_hot(fl_nm)
    output=netG2.forward(one_hot.view(1,-1,1,1))
    new_img_PIL = transforms.ToPILImage()(output[0])
    new_img_PIL.save(args.results2 + current_time + "--" + str(args.loss) + "/" + fl_nm[0] + ".png")
    im=Image.imread(image)
    im = torch.from_numpy(im)
    im = im.view(1,1,64,448)

    ll=loss(im.float(),output[0])*255
    total_loss=total_loss+ll
    # print("loss",ll)
    # break
    i=i+1
    if i>100:
        break

with open(args.results2 + current_time + "--" + str(args.loss) + "/" + "args.txt", "w") as f:
    f.write(str(args))
with open(args.results2 + current_time + "--" + str(args.loss) + "/" + "network.txt", "w") as f:
    f.write(str(netG2))
print("total_loss",total_loss)
print("ave_loss",total_loss/100) # to show the quality