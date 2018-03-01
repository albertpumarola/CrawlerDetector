import torch.nn as nn
from .networks import NetworkBase

class SmallNet(NetworkBase):

    def __init__(self, freeze=False):
        super(SmallNet, self).__init__()

        self._features = self._make_fatures()
        self._bb_reg = self._make_reg(1)
        self._prob_reg = self._make_reg(4)

        self._set_requires_grads(self._features, requires_grads=(not freeze))
        self._set_requires_grads(self._bb_reg, requires_grads=(not freeze))
        self._set_requires_grads(self._prob_reg, requires_grads=(not freeze))



    def _make_fatures(self):
        cfg = [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M', 512, 'M', 512, 'M']
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def _make_reg(self, out_nc):
        return nn.Sequential(
            nn.Linear(512*2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, out_nc),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self._features(x)
        features = features.view(features.size(0), -1)
        return self._bb_reg(features), self._prob_reg(features)