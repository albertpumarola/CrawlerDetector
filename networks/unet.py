import torch.nn as nn
from .networks import NetworkBase
import torch
import torch.nn.functional as F

class UNet(NetworkBase):
    def __init__(self, input_nc=3, output_nc=1, max_nc=512, min_nc=32, inplace=True, norm_type="instance"):
        super(UNet, self).__init__()

        norm_layer = self._get_norm_layer(norm_type)
        activation = nn.LeakyReLU(0.2, inplace=inplace)
        conv = nn.Conv2d

        self._name = "UNet"
        self._conv1 = nn.Sequential(*[conv(input_nc, 32, 7, stride=1, padding=3), norm_layer(32), activation])
        self._conv2 = nn.Sequential(*[conv(32, 32, 7, stride=1, padding=3), norm_layer(32), activation])
        self._conv3 = nn.Sequential(*[conv(32+input_nc, 32, 3, stride=1, padding=1), norm_layer(32), activation])
        self._conv4 = nn.Sequential(*[conv(32, output_nc, 3, stride=1, padding=1)])

        n_down = 5
        in_ch = 32
        for i in range(n_down):
            cur_in_ch = max(min(in_ch, max_nc), min_nc)
            cur_out_ch = max(min(2*in_ch, max_nc), min_nc)
            setattr(self, '_down_conv' + str(i), DownConvS(cur_in_ch, cur_out_ch, 3, conv, norm_layer, activation))
            in_ch *= 2

        for i in range(n_down):
            cur_in_ch = max(min(in_ch, max_nc), min_nc)
            cur_out_ch = max(min(in_ch//2, max_nc), min_nc)
            setattr(self, '_up_conv' + str(i), UpConvS(cur_in_ch, cur_out_ch, conv, norm_layer, activation))
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
    def __init__(self, input_nc, output_nc, k_size, conv, norm_layer, activation):
        super(DownConvS, self).__init__()

        model = [nn.AvgPool2d(2)]
        model += [conv(input_nc, output_nc, k_size, stride=1, padding=int((k_size - 1) / 2)),
                  norm_layer(output_nc),
                  activation]
        model += [conv(output_nc, output_nc, k_size, stride=1, padding=int((k_size - 1) / 2)),
                  norm_layer(output_nc),
                  activation]

        self._down_conv = nn.Sequential(*model)

    def forward(self, x):
        return self._down_conv(x)


class UpConvS(nn.Module):
    def __init__(self, input_nc, output_nc, conv, norm_layer, activation):
        super(UpConvS, self).__init__()

        self._conv1 = nn.Sequential(*[conv(input_nc, output_nc, 3, stride=1, padding=1),
                                      norm_layer(output_nc),
                                      activation])

        self._conv2 = nn.Sequential(*[conv(2 * output_nc, output_nc, 3, stride=1, padding=1),
                                      norm_layer(output_nc),
                                      activation])

    def forward(self, x, s):
        y = self._conv1(F.interpolate(x, scale_factor=2, mode='bilinear'))
        return self._conv2(torch.cat((y, s), 1))

class UpConv(nn.Module):
    def __init__(self, input_nc, output_nc, conv, norm_layer, activation):
        super(UpConv, self).__init__()

        self._conv1 = nn.Sequential(*[conv(input_nc, output_nc, 3, stride=1, padding=1),
                                      norm_layer(output_nc),
                                      activation])

    def forward(self, x):
        return self._conv1(F.interpolate(x, scale_factor=2, mode='bilinear'))
