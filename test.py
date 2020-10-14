from models import Generator
import torch
import random
from args import get_args
from torchvision import transforms
dit={"0":0,"A":1,"B":2,"C":3,"D":4,"E":5}
dic={ 0 :"0",1: "A",2: "B",3: "C",4: "D",5: "E"}
def str_to_one_hot(str):
    lens=len(str)
    # print(lens)
    aa=torch.zeros(lens,42)
    for i in range(0,lens):
      new_t_vecs = torch.zeros(7, 6)
      for j in range(0,7):
        new_t_vecs[j][dit[str[i][j]]]=1
        aa1=new_t_vecs.view(1,-1)
        aa[i]=aa1
    return aa

args=get_args()
ngpu=1
netG2=Generator(ngpu)
netG2.load_state_dict(torch.load(args.save_model+'params.pkl'))
# str=["AAAAAAA"]
# out=str_to_one_hot(str).view(1,-1,1,1)
# out2=netG2.forward(out)
# from torchvision import transforms
#
# new_img_PIL = transforms.ToPILImage()(out2[0])
# new_img_PIL.show() # 处理后的PIL图片
# new_img_PIL.save("1.png")

for i in range(0,20):
    fl_nm = ""
    for j in range(0,7):
       pic_nm=random.randint(0, 5)
       print(pic_nm)
       fl_nm = fl_nm + dic[pic_nm]
    print(fl_nm)
    fl_nm=[fl_nm]
    one_hot=str_to_one_hot(fl_nm)
    output=netG2.forward(one_hot.view(1,-1,1,1))
    new_img_PIL = transforms.ToPILImage()(output[0])
    new_img_PIL.save(args.results + fl_nm[0]+".png")
