import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from .trainers import HMDB51Trainer, UCF101Trainer
from .ClockworkRNN import ClockworkRNN
from .DilatedRNN import DRNN
from .Transformer_Encoder import Transformer_Encoder
from .AttentionRNN import AttentionRNN
from .fusion import EarlyFusedConv, SlowFusedConv, LateFusedConv

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv3d:
        torch.nn.init.kaiming_uniform_(m.weight)
        
class nop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x): return x
    
def linear(nh, no):
    return nn.Sequential(nn.Linear(nh,no),nn.ReLU())

class HMDB51_ConvRecurrent(HMDB51Trainer):
    def __init__(self, conv="vanilla", rnn="RNN",
                 rnn_args=(8192, 1024, 1),
                 drnn_args=(8192, 1024//4, 4),
                 attention_args = (8192, 1024),
                 nc=51, nf = 50,
                 pretrained=False, **kwargs):
        super(ConvRecurrent,self).__init__(**kwargs)
        nh = 8192
        
        if conv == "EF":
            self.conv = fusion.EarlyFusedConv(pretrained=pretrained)
            self.use_rnn = False
        elif conv == "SF":
            self.conv = fusion.SlowFusedConv(pretrained=pretrained)
            self.use_rnn = False
        elif conv == "LF":
            self.conv = fusion.LateFusedConv(pretrained=pretrained)
            self.use_rnn = False
        else:
            self.conv = models.resnet50(pretrained=pretrained)
            self.conv.avgpool = nop()
            self.conv.fc = nop()
            self.nf = 50
            nh = 1024
            sl = 50
            self.use_rnn = True
            self.gated = False
            self.dilation = False
            self.attention = False
        
        if self.use_rnn:
            if rnn == "RNN":
                assert(rnn_args!=None)
                self.rnn = nn.RNN(*rnn_args)
            elif rnn == "GRU":
                assert(rnn_args!=None)
                self.rnn = nn.GRU(*rnn_args)
            elif rnn == "LSTM":
                assert(rnn_args!=None)
                self.rnn = nn.LSTM(*rnn_args)
                self.gated = True
            elif rnn == "CRNN":
                clock_periods = [2**i for i in range(5)]
                self.rnn = ClockworkRNN(8192, 10, 512, clock_periods, **kwargs)
                nh = 50
            elif rnn == "DRNN":
                assert(drnn_args!=None)
                self.rnn = DRNN(*drnn_args)
                self.dilation =True
                nh = 2048
            elif rnn == "Attention":
                assert(rnn_args!=None)
                self.rnn = AttentionRNN(*attention_args, sl=sl)
                self.attention = True
                nh = 10240
            elif rnn == "Transformer":
                self.rnn = Transformer_Encoder(2048,self.nf,512,n=4,scale=8)
            else:
                raise NameError()
            
        self.fc = nn.Sequential(linear(int(nh),1024),
                                linear(1024,512),
                                linear(512,256),
                                linear(256,128),
                                nn.Linear(128,nc))
        
    def forward(self, X, verbose = False, collect=False):
        #X = X.transpose(-3,-1)
        bs, nf, c, h, w = X.shape
        X = X.view(bs*nf,c,h,w)
        emb = self.conv(X)
        if collect: self.embeddings.append(emb.reshape(-1,512,2,2).clone().detach()) #collect embeddings
        if self.use_rnn == True:
            emb = emb.view(bs,self.nf,-1).transpose(0,1)
            if self.gated:
                emb = self.rnn(emb)[1][0].transpose(0,1)
            elif self.dilation:
                emb = self.rnn(emb)[1][-1].transpose(0,1)
            elif self.attention:
                emb = self.rnn(emb)[0].transpose(0,1)
            else:
                emb = self.rnn(emb)[1].transpose(0,1)
            emb = emb.flatten(1)
        else:
            emb = emb.view(bs,-1)
        return torch.softmax(self.fc(emb), dim = 0)

class UCF101_ConvRecurrent(UCF101Trainer):
    def __init__(self, conv="vanilla", rnn="RNN",
                 rnn_args=(8192, 1024, 1),
                 drnn_args=(8192, 1024//4, 4),
                 attention_args = (8192, 1024),
                 nc=101, nf = 50,
                 pretrained=False, **kwargs):
        super(ConvRecurrent,self).__init__(**kwargs)
        nh = 8192
        
        if conv == "EF":
            self.conv = fusion.EarlyFusedConv(pretrained=pretrained)
            self.use_rnn = False
        elif conv == "SF":
            self.conv = fusion.SlowFusedConv(pretrained=pretrained)
            self.use_rnn = False
        elif conv == "LF":
            self.conv = fusion.LateFusedConv(pretrained=pretrained)
            self.use_rnn = False
        else:
            self.conv = models.resnet50(pretrained=pretrained)
            self.conv.avgpool = nop()
            self.conv.fc = nop()
            self.nf = 50
            nh = 1024
            sl = 50
            self.use_rnn = True
            self.gated = False
            self.dilation = False
            self.attention = False
        
        if self.use_rnn:
            if rnn == "RNN":
                assert(rnn_args!=None)
                self.rnn = nn.RNN(*rnn_args)
            elif rnn == "GRU":
                assert(rnn_args!=None)
                self.rnn = nn.GRU(*rnn_args)
            elif rnn == "LSTM":
                assert(rnn_args!=None)
                self.rnn = nn.LSTM(*rnn_args)
                self.gated = True
            elif rnn == "CRNN":
                clock_periods = [2**i for i in range(5)]
                self.rnn = ClockworkRNN(8192, 10, 512, clock_periods, **kwargs)
                nh = 50
            elif rnn == "DRNN":
                assert(drnn_args!=None)
                self.rnn = DRNN(*drnn_args)
                self.dilation =True
                nh = 2048
            elif rnn == "Attention":
                assert(rnn_args!=None)
                self.rnn = AttentionRNN(*attention_args, sl=sl)
                self.attention = True
                nh = 10240
            elif rnn == "Transformer":
                self.rnn = Transformer_Encoder(2048,self.nf,512,n=4,scale=8)
            else:
                raise NameError()
            
        self.fc = nn.Sequential(linear(int(nh),1024),
                                linear(1024,512),
                                linear(512,256),
                                linear(256,128),
                                nn.Linear(128,nc))
        
    def forward(self, X, verbose = False, collect=False):
        #X = X.transpose(-3,-1)
        bs, nf, c, h, w = X.shape
        X = X.view(bs*nf,c,h,w)
        emb = self.conv(X)
        if collect: self.embeddings.append(emb.reshape(-1,512,2,2).clone().detach()) #collect embeddings
        if self.use_rnn == True:
            emb = emb.view(bs,self.nf,-1).transpose(0,1)
            if self.gated:
                emb = self.rnn(emb)[1][0].transpose(0,1)
            elif self.dilation:
                emb = self.rnn(emb)[1][-1].transpose(0,1)
            elif self.attention:
                emb = self.rnn(emb)[0].transpose(0,1)
            else:
                emb = self.rnn(emb)[1].transpose(0,1)
            emb = emb.flatten(1)
        else:
            emb = emb.view(bs,-1)
        return torch.softmax(self.fc(emb), dim = 0)