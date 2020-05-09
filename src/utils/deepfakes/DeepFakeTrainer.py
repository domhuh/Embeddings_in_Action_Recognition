import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .OneCycleLearning import *
from tqdm.notebook import tqdm

class DeepFakeTrainer(nn.Module):
    def __init__(self, nf, criterion=nn.BCELoss()):
        super().__init__()
        self.criterion=criterion
        self.training_loss, self.validation_loss = [], []
        self.training_accuracy, self.validation_accuracy = [0.0],[0.0]
        self.nf = nf
        
    def fit(self, tdl, vdl=None, epochs = 1, lr = (1e-4,1e-3), ne=5):
        optimizer = optim.RMSprop(self.parameters(), lr=lr[0], weight_decay=0.01, momentum=0.9)
        scheduler = OneCycleLR(optimizer, num_steps=5, lr_range=lr)
        pb = tqdm(range(epochs))
        for epoch in pb:
            for video, label in tdl:
                self.train()
                optimizer.zero_grad()
                self.half()
                pred = self(video.cuda().half())
                self.float()
                loss = self.criterion(pred.float().squeeze(), label.cuda())
                loss.backward()
                scheduler.step()
                pb.set_description(f"""  Loss: {round(loss.item(),2)} || 
                                         Train: {round(np.mean(self.training_accuracy),2)}% ||
                                         Valid: {round(np.mean(self.validation_accuracy),2)}% """)
            self.evaluate(tdl,training=True)
            if not vdl is None: self.evaluate(vdl,validation=True)

    
    def evaluate(self, dl, training=False, validation=False):
        self.eval()
        correct, total = 0.0, 0.0
        losses = []
        with torch.no_grad():
            for video, label in dl:
                pred = self(video.cuda())
                rounded = ((torch.tensor([0.5]).expand(pred.shape[0],1).cuda() < pred)*1.0).squeeze()
                correct += sum(rounded == label.cuda()).item()
                total+=label.shape[0]
                losses.append(self.criterion(pred.squeeze(), label.cuda()).item())
            if training:
                self.training_accuracy.append(correct/total * 100)
                self.training_loss.append(np.mean(losses))
            elif validation:
                self.validation_accuracy.append(correct/total * 100)
                self.validation_loss.append(np.mean(losses))
        return correct/total