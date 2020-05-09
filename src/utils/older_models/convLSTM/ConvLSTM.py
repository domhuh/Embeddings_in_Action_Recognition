from utils import *
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from efficientnet_pytorch import EfficientNet 

class ConvLSTM(nn.Module):
    def __init__(self, hidden_dim = 1, num_layers =1,
                 dropout = 0):
        super().__init__()
        self.conv = EfficientNet.from_pretrained('efficientnet-b7')
        self.gru = nn.GRU(2560 * 4 * 4, hidden_dim, num_layers, dropout = dropout)
        self.fc = nn.Linear(hidden_dim,1)
        self.losses, self.accuracy = [], []
        
    def forward(self, X, verbose = False):
        init = True
        for seq in X:
            fm = self.conv.extract_features(seq)
            enc = torch.flatten(fm, start_dim = 1).unsqueeze(0) if init else torch.cat([enc, torch.flatten(fm, start_dim = 1).unsqueeze(0)], dim = 0)
            init = False
        x = self.gru(enc.transpose(0,1))[1]
        return torch.flatten(torch.sigmoid(self.fc(x)), start_dim=0)

    def fit(self, X, y, epochs = 1): #train on entire video
        criterion=nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        pb = tqdm(range(epochs))
        for epoch in pb:
            correct, total = 0.0, 0.0
            for i, data in enumerate(zip(X,y)):
                inputs, labels = data
                for x in torch.split(inputs,10):
                    self.train()
                    pred = self(x[None,:])
                    loss = criterion(pred, labels.unsqueeze(0))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.losses.append(loss.item())
                    if (torch.tensor([0.5]).cuda() < pred)*1 == labels:
                        correct+=1
                    total+=1
                    pb.set_description(f"""{epoch} ||
                                       Loss: {round(np.mean(np.array(self.losses[-10:])),3)} || 
                                       Accuracy: {round(correct/total * 100,2)} %
                                       """)
            self.accuracy.append(correct/total)

    
    def fit_(self, X, Y, epochs = 1): #train on single random sample per video
        criterion=nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        pb = tqdm(range(epochs))
        for epoch in pb:
            correct, total = 0.0, 0.0
            for i, data in enumerate(zip(X,Y)):
                self.train()
                x, y = data
                xs = torch.split(x,10)
                pred = self(xs[random.randint(0,len(xs)-1)][None,:])
                loss = criterion(pred, y.unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.losses.append(loss.item())
                if (torch.tensor([0.5]).cuda() < pred)*1 == y:
                    correct+=1
                total+=1
                pb.set_description(f"""{epoch} ||
                                   Loss: {round(np.mean(np.array(self.losses[-10:])),2)} || 
                                   Accuracy: {round(correct/total * 100,2)} %
                                   """)
            self.accuracy.append(correct/total)

    
    def acc(self, x, y):
        correct, total = 0.0, 0.0
        for xs,ys in zip(x,y):
            for xss in torch.split(xs,10):
                pred = (torch.tensor([0.5]).cuda() < self(xss[None,:]))*1
                if pred == ys:
                    correct += 1
                total += 1
        return (correct/total)