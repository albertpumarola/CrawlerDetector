import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch
import torch.utils.model_zoo as model_zoo

class uvProbNet(NetworkBase):
    def __init__(self, num_nc=32, do_add_batchnorm=False):
        super(uvProbNet, self).__init__()
        self._name = 'uvProbNet'

        features_cfg = [num_nc, num_nc, 'M', 2*num_nc, 2*num_nc, 'M', 4*num_nc, 4*num_nc,  'M', 6*num_nc, 6*num_nc,  'M', 6*num_nc, 6*num_nc, 'M']

        self._features = self._make_layers(features_cfg, 3, do_add_batchnorm)
        # self._pose_conv = self._make_layers(pose_cfg, 4*num_nc, batch_norm=False)
        # self._prob_conv = self._make_layers(prob_cfg, 4*num_nc, batch_norm=False)

        self._pose_reg = self._make_reg(2, 6*num_nc)
        self._prob_reg = self._make_reg(1, 6*num_nc)
        self.init_weights(self)

    def _make_layers(self, cfg, in_channels, batch_norm):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(0.2, inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(0.2, inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _make_reg(self, num_classes, nc):
        return nn.Sequential(
            nn.Linear(nc * 7 * 7, nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Linear(nc, nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(),
            nn.Linear(nc, num_classes),
        )

    def forward(self, x):
        features = self._features(x)
        pose = self._pose_reg(features.view(x.size(0), -1))
        prob = self._prob_reg(features.view(x.size(0), -1))
        return pose, prob


class AddCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, **kwargs):
        super().__init__()
        self.addcoords = AddCoords()
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size, out_channels, kernel, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret