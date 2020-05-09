import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset

class HMDB51Trainer(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(), device = 'cuda'):
        super().__init__()
        self.criterion=criterion
        self.training_loss, self.validation_loss = [], []
        self.training_accuracy, self.validation_accuracy = [0.0],[0.0]
        self.embeddings = []
        self.reduced_embeddings = []
        self.device = device
        
    def fit(self, tdl, vdl=None, epochs = 1, lr = 1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        pb = tqdm(range(epochs))
        for epoch in pb:
            for video, label in tdl:
                self.train()
                optimizer.zero_grad()
                pred = self(video.to(self.device).float())#.half())
                loss = self.criterion(pred.float().squeeze(), label.to(self.device))
                loss.backward()
                optimizer.step()
                pb.set_description(f"""  Loss: {round(loss.item(),2)} || 
                                         Train: {round(np.mean(self.training_accuracy),2)}% ||
                                         Valid: {round(np.mean(self.validation_accuracy),2)}% """)
            self.evaluate(tdl,training=True)
            if not vdl is None: self.evaluate(vdl,validation=True)
            self.embeddings = torch.cat(self.embeddings)
            if epoch == 0: self.create_kernel(self.embeddings.shape[0])# min(self.embeddings.shape))
            with torch.no_grad():
                self.reduced_embeddings.append(self.kernel(self.embeddings[None,:]).squeeze())
            self.embeddings = []
    
    def evaluate(self, dl, training=False, validation=False):
        self.eval()
        self.float()
        correct, total = 0.0, 0.0
        losses = []
        dl = DataLoader(dl.dataset, shuffle=False, batch_size = 128)
        with torch.no_grad():
            for video, label in dl:
                pred = self(video.to(self.device).float(), collect=training)
                correct += sum(torch.argmax(pred, dim=1) == label.to(self.device)).item()
                total+=label.shape[0]
                losses.append(self.criterion(pred.float().squeeze(), label.to(self.device)).item())
            if training:
                self.training_accuracy.append(correct/total * 100)
                self.training_loss.append(np.mean(losses))
            elif validation:
                self.validation_accuracy.append(correct/total * 100)
                self.validation_loss.append(np.mean(losses))
        return correct/total
    
        
    def create_kernel(self, ni, nf=1):
        self.kernel = nn.Conv3d(ni,1,nf).to(self.device)#,padding=(0,nf//2,nf//2))
        torch.nn.init.kaiming_uniform_(self.kernel.weight.data)
        self.kernel.bias.data.fill_(0.0)

class UCF101Trainer(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(), device = 'cuda'):
        super().__init__()
        self.criterion=criterion
        self.training_loss, self.validation_loss = [], []
        self.training_accuracy, self.validation_accuracy = [0.0],[0.0]
        self.embeddings = []
        self.reduced_embeddings = []
        self.device = device
        
    def fit(self, tdl, vdl=None, epochs = 1, lr = 1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        pb = tqdm(range(epochs))
        for epoch in pb:
            for video, label in tdl:
                self.train()
                optimizer.zero_grad()
                pred = self(video.to(self.device).float())#.half())
                loss = self.criterion(pred.float().squeeze(), label.to(self.device))
                loss.backward()
                optimizer.step()
                pb.set_description(f"""  Loss: {round(loss.item(),2)} || 
                                         Train: {round(np.mean(self.training_accuracy),2)}% ||
                                         Valid: {round(np.mean(self.validation_accuracy),2)}% """)
            self.evaluate(tdl,training=True)
            if not vdl is None: self.evaluate(vdl,validation=True)
            self.embeddings = torch.cat(self.embeddings)
            if epoch == 0: self.create_kernel(self.embeddings.shape[0])# min(self.embeddings.shape))
            with torch.no_grad():
                self.reduced_embeddings.append(self.kernel(self.embeddings[None,:]).squeeze())
            self.embeddings = []
    
    def evaluate(self, dl, training=False, validation=False):
        self.eval()
        self.float()
        correct, total = 0.0, 0.0
        losses = []
        dl = DataLoader(dl.dataset, shuffle=False, batch_size = 128)
        with torch.no_grad():
            for video, label in dl:
                pred = self(video.to(self.device).float(), collect=training)
                correct += sum(torch.argmax(pred, dim=1) == label.to(self.device)).item()
                total+=label.shape[0]
                losses.append(self.criterion(pred.float().squeeze(), label.to(self.device)).item())
            if training:
                self.training_accuracy.append(correct/total * 100)
                self.training_loss.append(np.mean(losses))
            elif validation:
                self.validation_accuracy.append(correct/total * 100)
                self.validation_loss.append(np.mean(losses))
        return correct/total
    
        
    def create_kernel(self, ni, nf=1):
        self.kernel = nn.Conv3d(ni,1,nf).to(self.device)#,padding=(0,nf//2,nf//2))
        torch.nn.init.kaiming_uniform_(self.kernel.weight.data)
        self.kernel.bias.data.fill_(0.0)