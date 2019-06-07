import torch.nn as nn
from .networks import NetworkBase
import torch
import torch.nn.functional as F

class UNet(NetworkBase):
    def __init__(self, input_nc=3, output_nc=1, max_nc=512, min_nc=32):
        super(UNet, self).__init__()

        # norm_layer = self._get_norm_layer(norm_type)
        # activation = nn.ReLU(inplace=inplace)
        # conv = nn.Conv2d

        self._name = "UNet"
        self._conv1 = nn.Sequential(*[nn.Conv2d(input_nc, 32, 7, stride=1, padding=3),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)])
        self._conv2 = nn.Sequential(*[nn.Conv2d(32, 32, 7, stride=1, padding=3),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)])
        self._conv3 = nn.Sequential(*[nn.Conv2d(32+input_nc, 32, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True)])
        self._conv4 = nn.Sequential(*[nn.Conv2d(32, output_nc, 3, stride=1, padding=1)])

        n_down = 5
        in_ch = 32
        for i in range(n_down):
            cur_in_ch = max(min(in_ch, max_nc), min_nc)
            cur_out_ch = max(min(2*in_ch, max_nc), min_nc)
            setattr(self, '_down_conv' + str(i), DownConvS(cur_in_ch, cur_out_ch, 3))
            in_ch *= 2

        for i in range(n_down):
            cur_in_ch = max(min(in_ch, max_nc), min_nc)
            cur_out_ch = max(min(in_ch//2, max_nc), min_nc)
            setattr(self, '_up_conv' + str(i), UpConvS(cur_in_ch, cur_out_ch))
            in_ch //= 2

        self.init_weights(self)

    def forward(self, x):
        s1 = self._conv2(self._conv1(x))
        s2 = self._down_conv0(s1)
        s3 = self._down_conv1(s2)
        s4 = self._down_conv2(s3)
        s5 = self._down_conv3(s4)
        y = self._down_conv4(s5)
        y = self._up_conv0(y, s5)
        y = self._up_conv1(y, s4)
        y = self._up_conv2(y, s3)
        y = self._up_conv3(y, s2)
        y = self._up_conv4(y, s1)
        y = self._conv3(torch.cat([y, x], -3))
        return self._conv4(y)

class DownConvS(nn.Module):
    def __init__(self, input_nc, output_nc, k_size):
        super(DownConvS, self).__init__()

        model = [nn.AvgPool2d(2)]
        model += [nn.Conv2d(input_nc, output_nc, k_size, stride=1, padding=int((k_size - 1) / 2)),
                  nn.BatchNorm2d(output_nc),
                  nn.ReLU(inplace=True)]

        self._down_conv = nn.Sequential(*model)

    def forward(self, x):
        return self._down_conv(x)


class UpConvS(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(UpConvS, self).__init__()

        self._conv2 = nn.Sequential(*[nn.Conv2d(input_nc + output_nc, output_nc, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(output_nc),
                                      nn.ReLU(inplace=True)])

    def forward(self, x, s):
        return self._conv2(torch.cat((F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False), s), 1))