import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_layers(start, hs, end, step):
    lse = [*list(range(hs, end, -step)), end]
    return list(zip([start,*lse[:]], [*lse[:], end]))[:-1]

class MLP(nn.Module):
    def __init__(self, conv, ls):
        super().__init__()
        self.conv = conv
        self.model = nn.Sequential(*[nn.Sequential(nn.Linear(*n), nn.ReLU()) for n in ls])
        self.sep = []
        
    def forward(self, X):
        return self.model(self.conv(X).flatten(1))

    def fit(self, tdl, vdl, lr = 1e-3):
    	self.train()
    	crit = nn.CrossEntropyLoss()
        op = optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            torch.cuda.empty_cache() if torch.cuda.device_count() else None
            for data in tdl:
                op.zero_grad()
                pred = self(data[0])
                loss = crit(pred, data[1])
                loss.backward()
                op.step()

        		self.sep.append(acc(pred,data[1])).item()

	def acc(self, out, Y):
        return (torch.argmax(out, dim=1)==Y.long()).float().mean()