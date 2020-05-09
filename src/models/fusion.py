import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models.utils import load_state_dict_from_url

class DWConv3d(nn.Module):
    def __init__(self, ni, nf, ks=3, n = 4, DW=True):
        super().__init__()
        try: assert(nf%n == 0)
        except: print(f"nf is {nf} and n is {n}")
        self.ni = ni
        self.nf = nf
        self.n = n
        if DW==True:
            self.conv = nn.Sequential(nn.Conv3d(ni,nf,(ks,1,1), padding = ks//2),
                                      nn.Conv3d(nf,nf,(1,ks,ks),groups=n))
        else:
            self.conv = nn.Conv3d(ni,nf,ks, padding = ks//2)
        self.pool = nn.MaxPool3d((3,1,1),(2,1,1),padding=(1,0,0)) #only pool temporal
        self.bn = nn.BatchNorm3d(nf//2)
        
    def forward(self,x):
        bs, nc, nh, nw = x.shape
        bs = bs//self.ni
        x = x.reshape(bs, self.ni, nc, nh, nw)
        x = self.conv(x).transpose(1,2) #(128,3,nf,64,64)
        x = self.pool(x).transpose(1,2) #switch back
        x = self.bn(x)
        x = torch.relu(x)
        return x.reshape(bs * self.nf//2, nc, nh, nw)

class EarlyFusedConv(nn.Module):
    def __init__(self, ni = 10, nf = 2, pretrained=False, DW = True):
        super(EarlyFusedConv,self).__init__()
        self.conv2d = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                                  progress=True)
            self.conv2d.load_state_dict(state_dict)
        self.conv3d = nn.ModuleList([DWConv3d(ni,   nf,   n=1, DW=DW),
                                     DWConv3d(nf//2,nf*2, n=1, DW=DW),
                                     DWConv3d(nf,   nf*4, n=1, DW=DW)])
    def forward(self,x):
        x = self.conv3d[0](x)        
        x = self.conv3d[1](x)        
        x = self.conv3d[2](x)        
        x = self.conv2d.conv1(x)
        x = self.conv2d.bn1(x)
        x = self.conv2d.relu(x)
        x = self.conv2d.maxpool(x)
        x = self.conv2d.layer1(x)
        x = self.conv2d.layer2(x)
        x = self.conv2d.layer3(x)
        x = self.conv2d.layer4(x)
        return x    

class SlowFusedConv(nn.Module):
    def __init__(self, ni = 10, nf = 2, pretrained=False, DW=True):
        super(SlowFusedConv,self).__init__()
        self.conv2d = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                                  progress=True)
            self.conv2d.load_state_dict(state_dict)
        self.conv3d = nn.ModuleList([DWConv3d(ni,   nf,   n=1, DW=DW),
                                     DWConv3d(nf//2,nf*2, n=1, DW=DW),
                                     DWConv3d(nf,   nf*4, n=1, DW=DW)])
    def forward(self,x):
        x = self.conv2d.conv1(x)
        x = self.conv2d.bn1(x)
        x = self.conv2d.relu(x)
        x = self.conv2d.maxpool(x)
        x = self.conv2d.layer1(x)
        x = self.conv3d[0](x)        
        x = self.conv2d.layer2(x)
        x = self.conv3d[1](x)
        x = self.conv2d.layer3(x)
        x = self.conv3d[2](x)
        x = self.conv2d.layer4(x)
        return x

class LateFusedConv(nn.Module):
    def __init__(self, ni = 10, nf = 2, pretrained=False, DW=True):
        super(LateFusedConv,self).__init__()
        self.conv2d = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth',
                                                  progress=True)
            self.conv2d.load_state_dict(state_dict)
        self.conv3d = nn.ModuleList([DWConv3d(ni,   nf,   n=1, DW=DW),
                                     DWConv3d(nf//2,nf*2, n=1, DW=DW),
                                     DWConv3d(nf,   nf*4, n=1, DW=DW)])
    def forward(self,x):
        x = self.conv2d.conv1(x)
        x = self.conv2d.bn1(x)
        x = self.conv2d.relu(x)
        x = self.conv2d.maxpool(x)
        x = self.conv2d.layer1(x)
        x = self.conv2d.layer2(x)
        x = self.conv2d.layer3(x)
        x = self.conv2d.layer4(x)
        x = self.conv3d[0](x)        
        x = self.conv3d[1](x)        
        x = self.conv3d[2](x)        
        return x