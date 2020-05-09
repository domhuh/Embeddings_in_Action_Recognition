import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class ClockworkRNN(nn.Module):
    """
    Source code created by Prashanth V (Github: prash2512)
    Modified by Dom Huh (Github: domhuh)
    
    __init__(ni,nh,no,clock_periods)
        ni = input size
        nh = hidden state size
        no = output size
        clock period = list of clock periods
    forward(x)
        x.shape() = (seq_length,batch_size,input_dim)
        returns Ys (seq_length,batch_size,output_size), H (len(clock_period), batch_size)
    """
    def __init__(self, ni, nh, no, clock_periods, device='cpu'):
        super(ClockworkRNN, self).__init__()
        self.nclocks = len(clock_periods)
        self.ni = ni
        self.nh = nh
        self.no = no
        self.schedules = self.make_schedule(clock_periods)
        self.clock_periods = clock_periods
        self.device = device
        self._build()
    
    def _build(self):
        Wi = self.glorotize(np.random.randn(self.nclocks * self.nh, self.ni + 1))
        Wh = np.random.randn(self.nclocks * self.nh, self.nclocks * self.nh + 1)
        Wo = self.glorotize(np.random.randn(self.no, self.nclocks * self.nh + 1))
        Wh[:, :-1] = self.orthogonalize(Wh[:, :-1])        
        Wh[:,:-1] *= self.recurrent_mask
            
        self.Wi = nn.Parameter(torch.from_numpy(Wi).float()).to(self.device)
        self.Wh = nn.Parameter(torch.from_numpy(Wh).float()).to(self.device)
        self.Wo = nn.Parameter(torch.from_numpy(Wo).float()).to(self.device)
        
    def forward(self, X):
        X = X.transpose(1,2)
        T, n, input_dim = X.size()
        Ys = torch.empty((T, self.no, input_dim)).to(self.device)
        H_prev = Variable(torch.zeros((self.nclocks * self.nh, input_dim))).to(self.device)

        for t in range(T):
            active = [int(t%self.schedules[i]==0) for i in range(len(self.schedules))]
            active = Variable(torch.FloatTensor(active).view(-1,1)).to(self.device)
            inputn = torch.cat([X[t],Variable(torch.ones(1,input_dim)).to(self.device)],0)
            i_h = torch.mm(self.Wi,inputn)
            _H_prev = torch.cat([H_prev,Variable(torch.ones((1, input_dim))).to(self.device)],0)
            H_new = torch.tanh(i_h + torch.mm(self.Wh, _H_prev))
            H = active.expand_as(H_new)*H_new+(1-active).expand_as(H_prev)*H_prev
            _H = torch.cat([H, Variable(torch.ones((1, input_dim))).to(self.device)],0)
            H_prev = H
            Ys[t] = torch.tanh(torch.mm(self.Wo, _H))

        Ys = Ys.transpose(1,2)
        return Ys, H
    
    def glorotize(self, W):
        W *= np.sqrt(6)
        W /= np.sqrt(np.sum(W.shape))
        return W 
    
    def orthogonalize(self, W):
        W, _, _ = np.linalg.svd(W)
        return W
    
    @property
    def recurrent_mask(self):
        matrix = []
        for c in range(self.nclocks, 0, -1):
            zero_blocks = np.zeros((self.nh, self.nh * (self.nclocks - c)))		
            one_blocks = np.ones((self.nh, self.nh * (c)))
            matrix.append(np.concatenate([zero_blocks, one_blocks], axis=1))
        mask = np.concatenate(matrix, axis=0)
        return mask
    
    def make_schedule(self, clock_periods):
        sch = []
        for c in clock_periods:
            for i in range(self.nh):
                sch.append(c)
        return sch