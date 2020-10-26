from models import Generator
import torch
import random
from args import get_args
from datetime import datetime
from torchvision import transforms
import os
import cv2
dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":26}
dic={ 0 :"0",1: "A",2: "B",3: "C",4: "D",5: "E",6:"F",7:"G",8:"H",9:"I",10:"J",11:"K",12:"L",13:"M",14:"N",15:"O",16:"P",17:"Q",18:"R",19:"S",20:"T",21:"U",22:"V",23:"W",24:"X",25:"Y",26:"Z"}
def str_to_one_hot(str):
    lens=len(str)
    aa=torch.zeros(lens,189)
    # aa=torch.zeros(lens,42)
    for i in range(0,lens):
      new_t_vecs = torch.zeros(7, 27)
      # new_t_vecs = torch.zeros(7, 6)
      for j in range(0,7):
        new_t_vecs[j][dit[str[i][j]]]=1
        aa1=new_t_vecs.view(1,-1)
        aa[i]=aa1
    return aa

args=get_args()
ngpu=1
# netG2=Generator(ngpu)
# netG2.load_state_dict(torch.load(args.save_model+'params3.pkl'))

netG2 = torch.load(args.save_model+'netG26_anti_mse_0.pkl')

# usr for the test of generating a single title image

# str=["ONE0CAT"]
# ins_p=str_to_one_hot(str)
# int_p2=ins_p.view(1,-1,1,1)
# out2=netG2.forward(int_p2)
# print(out2.size())
# new_img_PIL = transforms.ToPILImage()(out2[0])
# new_img_PIL.show() #
# new_img_PIL.save("ONE0CAT.png")



current_time = datetime.now().strftime('%b%d_%H-%M-%S')
os.mkdir(args.results +current_time+"--"+str(args.char_num)+"--"+str(args.loss))

# generate 50 images as the test results
for i in range(0,50):
    fl_nm = ""
    for j in range(0,7):
       pic_nm=random.randint(0, 26)
       fl_nm = fl_nm + dic[pic_nm]
    fl_nm=[fl_nm]
    one_hot=str_to_one_hot(fl_nm)
    output=netG2.forward(one_hot.view(1,-1,1,1))

    new_img_PIL = transforms.ToPILImage()(output[0])
    new_img_PIL.save(args.results +current_time+"--"+str(args.char_num)+"--"+str(args.loss)+"/"+ fl_nm[0]+".png")

# save the parameters and the structure of the network
with open(args.results+current_time+"--"+str(args.char_num)+"--"+str(args.loss)+"/" +"args.txt","w") as f:
    f.write(str(args))
with open(args.results+current_time+"--"+str(args.char_num)+"--"+str(args.loss)+"/" +"network.txt","w") as f:
    f.write(str(netG2))