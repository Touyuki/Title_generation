import torch
import torch.nn as nn
ngf=32
class Generator(nn.Module):
   def __init__(self, ngpu):
      super(Generator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
          nn.ConvTranspose2d(189, ngf * 32, (1, 7), 1, 0, bias=False),
          # nn.ConvTranspose2d( 42, ngf * 32, (1,7), 1, 0, bias=False),
          nn.BatchNorm2d(ngf * 32),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf * 32, ngf * 16, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf * 16),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf * 8),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf * 8, ngf*4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf*4, ngf*2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
          nn.Tanh()
      )
   def forward(self, input):
        # x=self.main2(input)
        return self.main(input)
# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)