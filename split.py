from PIL import Image
import numpy as np
from args import get_args
args=get_args()

image = Image.open(args.font_address +"Antique Book Cover.0.0.png")
# image = Image.open(args.font_address +"Code New Roman.0.0.png")

image_arr = np.array(image)
for i in range(1,27):
    a = np.zeros((64,64),np.int8)
    a = image_arr[:,(i-1)*64:i*64]
    filename=str(i)
    nw=Image.fromarray(a.astype(np.uint8))
    nw.save(args.single_chr+"base/"+filename+".png")

# make the image of the space
b=np.zeros((64,64),np.int8)
b[:,:]=255
nw=Image.fromarray(b.astype(np.uint8))
nw.save(args.single_chr+"base/"+"0"+".png")