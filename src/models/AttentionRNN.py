import torch.nn as nn
import torch
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    """
    Fixed sequence length implementation of AttentionRNN
    """
    def __init__(self, ni, hs, sl=10, **kwargs):
        super().__init__(**kwargs)
        self.rnn = nn.RNN(ni, hs, **kwargs)
        self.key = nn.Sequential(nn.Linear(hs,hs),
                                 nn.LeakyReLU(),
                                 nn.Linear(hs,hs*sl))
        self.query = nn.Sequential(nn.Linear(ni*sl,hs),
                                   nn.LeakyReLU(),
                                   nn.Linear(hs,hs*sl))
        self.value = nn.Sequential(nn.Linear(hs*sl,hs),
                                   nn.LeakyReLU(),
                                   nn.Linear(hs,hs*sl))
        self.attnmap = None
        self.atfm = None
        
    def forward(self, X):
        o, h = self.rnn(X)
        
        xsl, xbs, xhs = X.shape
        hsl, hbs, hhs = h.shape
        osl, obs, ohs = o.shape

        key = F.relu(h.reshape(hbs, hsl*hhs))
        query = X.reshape(xbs, xsl*xhs)
        value = F.relu(o.reshape(obs, osl*ohs))

        key = self.key(key).reshape(obs,osl,-1).transpose(1,2)
        query = self.query(query).reshape(xbs,xsl,-1)
        value = self.value(value).reshape(obs,osl,-1)
        self.attnmap = torch.softmax(torch.bmm(key,query),dim=2)
        self.atfm = torch.bmm(value,self.attnmap).transpose(0,1)
        return self.atfm, self.attnmap
