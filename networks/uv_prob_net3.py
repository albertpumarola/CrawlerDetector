from .networks import NetworkBase
from torchvision import models
import torch.nn as nn
import torch

class uvProbNet(NetworkBase):
    def __init__(self):
        super(uvProbNet, self).__init__()

        self._name = "uvProbNet"
        self._vgg11 = VGG11()

        self._pos_reg = Regresor(512, 2, add_sigmoid=True)
        self._prob_reg = Regresor(512, 1)

    def forward(self, x):
        features = self._vgg11(x)
        features = features.view(x.size(0), -1)
        pos = self._pos_reg(features)
        prob = self._prob_reg(features)
        return pos, prob

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()

        self._vgg11 = models.vgg11(pretrained=True).features
        #
        # for param in self._vgg11.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self._vgg11(x)


class Regresor(NetworkBase):
    def __init__(self, input_nc, output_nc, add_sigmoid=False):
        super(Regresor, self).__init__()

        # self._avgpool = nn.AdaptiveAvgPool2d((4, 4))
        model = [nn.Linear(input_nc * 4 * 4, input_nc//2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(input_nc//2, input_nc//2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(input_nc//2, output_nc)]

        if add_sigmoid:
            model += [nn.Sigmoid()]

        self._reg = nn.Sequential(*model)

        self.init_weights(self)

    def forward(self, x):
        # features = self._avgpool(x).view(x.size(0), -1)
        return self._reg(x)

