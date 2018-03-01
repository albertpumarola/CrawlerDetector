from __future__ import absolute_import
import sys
sys.path.append('/home/apumarola/code/phd/Crawler-Detector/')

from options.train_options import TrainOptions
from torch.autograd import Variable
from networks.vgg_features import VggFeatures
from networks.prob_net import ProbNet
from networks.bb_net import BBNet
from networks.small_net import SmallNet

import torch

opt = TrainOptions().parse()

B, C, H, W = 4, 3, opt.image_size_h, opt.image_size_w
batch_imgs = Variable(torch.ones([B, C, H, W])).cuda()

# feature_extractor = VggFeatures()
# feature_extractor = feature_extractor.cuda()
#
# bb_net = BBNet()
# bb_net = bb_net.cuda()
#
# prob_net = ProbNet()
# prob_net.cuda()

# features = feature_extractor(batch_imgs)
# print('features', features.size())

# bb = bb_net(features)
# print('bb', bb.size())
#
# prob = prob_net(features)
# print('prob', prob.size())

net = SmallNet().cuda()
a, b = net(batch_imgs)
print a.size(), b.size()