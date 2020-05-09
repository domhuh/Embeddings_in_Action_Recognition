import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def linear(nh, no):
    return nn.Sequential(nn.Linear(nh,no),nn.ReLU())
class VAE(nn.Module):
    def __init__(self, conv, nh, nf=64, device = 'cpu'):
        super().__init__()
        self.conv = conv
        self.criterion = self.ELBO
        self.optimizer = optim.Adam
        self.mu = nn.Linear(256, nf)
        self.mu_encoder = nn.Sequential(linear(nh,512),
                                     linear(512,256),
                                     self.mu)
        self.var = nn.Linear(256, nf)
        self.var_encoder = nn.Sequential(linear(nh,512),
                             linear(512,256),
                             self.var)
        self.decoder = nn.Sequential(linear(nf, 256),
                                     linear(256,512),
                                     nn.Linear(512,nh))
        self.device = device
        self.training_losses = []
        self.validation_losses = []
    def encode(self, x):
        fm = x.flatten(1)#torch.flatten(self.conv(x), start_dim=1)
        mu = self.mu_encoder(fm)
        var = self.var_encoder(fm)
        return mu, var
    
    def reparameterize(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def get_dist(self, x):
        mu, var = self.encode(x)
        return self.reparameterize(mu, var)
        
    def decode(self, mu, var):
        z = self.reparameterize(mu, var)
        return torch.sigmoid(self.decoder(z))
    
    def get_feature_map(self, x):
        with torch.no_grad():
            fm= self.conv(x)
        return torch.flatten(fm,start_dim=1)   
    
    

    def fit_encoder(self, x, vdl=None,
                    num_epochs=1, lr = 1e-3, **kwargs):

        pb = tqdm(range(num_epochs))
        self.optim = self.optimizer([*self.mu_encoder.parameters(),*self.var_encoder.parameters(),*self.decoder.parameters()],
                                    lr=lr, **kwargs)
        for e in pb:
            self.train()
            losses = []
            #for x, _ in tdl:
            x = self.preprocess(x.to(self.device))
            self.optim.zero_grad()
            y = x#self.get_feature_map(x)
            mu, var = self.encode(x)
            yhat = self.decode(mu, var)
            loss = self.criterion(yhat,y, mu, var)
            loss.backward()
            losses.append(loss.item())
            self.optim.step()
            pb.set_description(f"Loss: {round(np.mean(losses),2)}")
                
            self.training_losses.append(np.mean(losses))
            if vdl is not None: self.evaluate(vdl, validation=True)
            

    def evaluate(self, dl, training=False, validation=False):
        self.eval()
        losses = []
        with torch.no_grad():
            for x, _ in dl:
                x = self.preprocess(x.to(self.device))
                y = self.get_feature_map(x)
                mu, var = self.encode(x)
                yhat = self.decode(mu, var)
                loss = F.binary_cross_entropy(yhat, y)
                losses.append(loss.item())
        if validation: self.validation_losses.append(np.mean(losses))
        if training: self.training_losses.append(np.mean(losses))
            
    def ELBO(self, yhat, y, mu, var):
        BCE = F.binary_cross_entropy(yhat, y)
        KLD = -0.5 * torch.sum(1 + var - mu**2 - var.exp())
        return BCE + KLD

    def preprocess(self,x):
        return x
        return x.reshape(-1,3,64,64).float()