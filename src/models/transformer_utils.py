import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import copy

def linear(nh, no):
    return nn.Sequential(nn.Linear(nh,no),nn.LeakyReLU())
    
"""
Code adapted from Harvard NLP
"""
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Norm(nn.Module):
    def __init__(self, nf, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(nf))
        self.b = nn.Parameter(torch.zeros(nf))
        self.eps = eps
    def forward(self,x):
        return self.a * (x-x.mean())/(x.std()+self.eps) + self.b

def mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query,key,value,mask=None,dropout=None):
    score = torch.bmm(query,key.transpose(-1,-2))/math.sqrt(query.size(-1))
    if mask is not None:
        score = score.masked_fill(mask == 0, 1e-9)
    attn = F.softmax(score, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.bmm(attn, value), attn

class MultiheadedAttention(nn.Module):
    def __init__(self, h, hs, dropout=0.0):
        super().__init__()
        self.h = h
        self.k = hs//h
        self.fc = clones(nn.Sequential(linear(hs,hs),
                                       linear(hs,hs),
                                       linear(hs,hs)),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.shape[0]
        query,key,value = [l(x).view(n_batch,self.h,self.k) for l,x in zip(self.fc,(query,key,value))]
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)
        return self.fc[-1](x.view(n_batch,-1))
        
class LayerBlock(nn.Module):
    def __init__(self,ni,nh,dropout=0.0):
        super().__init__()
        self.self_attention=MultiheadedAttention(8,ni)
        self.fc = PositionwiseFeedForward(ni,nh)
        self.norm_1 = Norm(ni)
        self.norm_2 = Norm(ni)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x_ = self.self_attention(x,x,x,mask)
        x =  x_ + self.dropout(self.norm_1(x))
        x_ = self.fc(x)
        return x_ + self.dropout(self.norm_2(x))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))