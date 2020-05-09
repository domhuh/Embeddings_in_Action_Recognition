from .transformer_utils import *
import torch.nn as nn
import torch

def linear(nh, no):
    return nn.Sequential(nn.Linear(nh,no),nn.LeakyReLU())

class Transformer_Encoder(nn.Module):
    def __init__(self,ni,nf,hs,n=3,dropout=0.0,scale=8):
        super().__init__()
        scaled = (ni*nf)//scale
        self.fan_in = nn.Sequential(nn.Linear(ni*nf,scaled),
                                    nn.Linear(scaled,scaled),
                                    nn.Linear(scaled,scaled))
        self.pe = PositionalEncoding(ni//scale,dropout)
        self.encoder = nn.Sequential(*[LayerBlock(scaled,hs,dropout=0) for _ in range(n)])
        self.fan_out =linear(scaled,hs*nf)
                                     
    def forward(self,x):
        x = x.transpose(0,1)
        n_batch, nf = x.shape[:2]
        x = torch.sigmoid(self.fan_in(x.flatten(1)).view(n_batch,nf,-1))
        x = self.pe(x).flatten(1)
        x = self.encoder(x).flatten(1)
        x = F.relu(self.fan_out(x))
        return x.view(n_batch,nf,-1).transpose(0,1), None