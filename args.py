import argparse

parser=argparse.ArgumentParser()

##address
parser.add_argument("--font_address",default="font/",type=str)
parser.add_argument("--datasets",default="test_data/",type=str)
parser.add_argument("--results",default="results/",type=str)
parser.add_argument("--single_chr",default="single/",type=str)
parser.add_argument("--save_model",default="weight_file/",type=str)

##hyperparameters
parser.add_argument("--num_epochs",default=20,type=int)
parser.add_argument("--length_tensor",default=42,type=int)
parser.add_argument("--ngf",default=32,type=int)
parser.add_argument("--lr",default=0.002,type=float)
parser.add_argument("--beta",default=0.5,type=float)
parser.add_argument("--batch_size",default=10,type=int)
parser.add_argument("--ngpu",default=1,type=int)

def get_args():
    return parser.parse_args()