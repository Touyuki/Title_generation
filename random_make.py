import numpy as np
from PIL import Image
from args import get_args
import random
args=get_args()
dic={ 0 :"0",1: "A",2: "B",3: "C",4: "D",5: "E",6: "F",7: "G",8: "H",9: "I",10: "J",11: "K",12: "L",13: "M",14: "N",15: "O",16: "P",17: "Q",18: "R",19: "S",20: "T",21: "U",22: "V",23: "W",24: "X",25: "Y",26: "Z"}
for i in range(0,24000):
    a = np.zeros((64, 448),np.int8)
    fl_nm=""
    for j in range(0,7):
        pic_nm=random.randint(0, 26) # 0:space 1:A ... 26:Z
        image = Image.open(args.single_chr+ str(pic_nm)+".png")  # create the training data with single letters
        image_arr = np.array(image)
        a[:,j*64:j*64+64]=image_arr
        fl_nm=fl_nm+dic[pic_nm]
    nw = Image.fromarray(a.astype(np.uint8))
    nw.save(args.datasets+"train/"+fl_nm+".png")
