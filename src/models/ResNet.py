import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(models.resnet.ResNet):
    """
    Rewritten to avoid torch.flatten from torchvision implementation
    """
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)
        
    def _forward_impl(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = models.resnet.load_state_dict_from_url(models.resnet.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, **kwargs):
    return _resnet('resnet18', models.resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress=True, **kwargs)