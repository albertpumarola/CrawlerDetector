import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import util.plots as plots
from .models import BaseModel
from networks.networks import NetworksFactory
import numpy as np
import torch.utils.model_zoo as model_zoo


class ObjectDetectorNetModel(BaseModel):
    def __init__(self, opt):
        super(ObjectDetectorNetModel, self).__init__(opt)
        self._name = 'ObjectDetectorNetModel'
        self._gpu_bb = 1 if len(self._gpu_ids) > 1 else 0  # everything related bb moved to gpu 1 that has more free mem

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()
        # else:
        #     self._load_vgg_weights()

        # prefetch variables
        self._init_prefetch_inputs()
        self._init_prefetch_create_hm_vars()

        # init
        self._init_losses()

    def set_input(self, input):
        # copy images efficiently
        pos_input_img = input['pos_img']
        neg_input_img = input['neg_img']
        self._pos_input_img.resize_(pos_input_img.size()).copy_(pos_input_img)
        self._neg_input_img.resize_(neg_input_img.size()).copy_(neg_input_img)

        # gt bb
        self._pos_gt = input['pos_norm_pose']
        pos = torch.unsqueeze(input['pos_norm_pose'], 1).cuda()
        self._pos_input_hm = self._gaussian_grid(pos, self._sigma, self._grid)

        # store paths
        self._pos_input_img_path = input['pos_img_path']
        self._neg_input_img_path = input['neg_img_path']

    def forward(self, keep_data_for_visuals=False, keep_estimation=False):
        # clean previous gradients
        if self._is_train:
           self._optimizer.zero_grad()

        # forward pass
        loss_pose = self._forward(keep_data_for_visuals, keep_estimation)

        # compute new gradients
        if self._is_train:
            loss_pose.backward()

    def test(self, image):
        pass
        # # bb as (top, left, bottom, right)
        # estim_bb_lowres, estim_prob = self._net(Variable(image, volatile=True))
        # bb = self._unormalize_bb(estim_bb_lowres.data.numpy()).astype(np.int)
        # prob = estim_prob.data.numpy()
        # return bb, prob

    def optimize_parameters(self):
        self._optimizer.step()

    def set_train(self):
        self._net.train()
        self._is_train = True

    def set_eval(self):
        self._net.eval()
        self._is_train = False

    def get_current_paths(self):
        return OrderedDict([('pos_img', self._pos_input_img_path),
                            ('neg_img', self._neg_input_img_path)])

    def get_current_errors(self):
        return OrderedDict([('pos_hm', self._loss_pos_hm.data[0]),
                            ('neg_hm', self._loss_neg_hm.data[0])])

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_last_saved_estimation(self):
        """
        Returns last model estimation with flag keep_estimation=True
        """
        return self._estim_dict

    def get_last_saved_visuals(self):
        """
        Returns last model visuals with flag keep_data_for_visuals=True
        """
        # visuals return dictionary
        visuals = OrderedDict()
        visuals['pos_gt_hm'] = plots.plot_overlay_attention(self._vis_input_pos_img, self._vis_gt_pos_hm[0, :, :])
        visuals['pos_estim_hm'] = plots.plot_overlay_attention(self._vis_input_pos_img, self._vis_estim_pos_hm[0, :, :])
        visuals['neg_gt_hm'] = plots.plot_overlay_attention(self._vis_input_pos_img, self._vis_gt_neg_hm[0, :, :])
        visuals['neg_estim_hm'] = plots.plot_overlay_attention(self._vis_input_neg_img, self._vis_estim_neg_hm[0, :, :])
        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._net, 'net', label)

        # save optimizers
        self._save_optimizer(self._optimizer, 'net', label)

    def load(self):
        # load networks
        self._load_network(self._net, 'net', self._opt.load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer, 'net', self._opt.load_epoch)

    def update_learning_rate(self):
        # updated learning rate bb net
        lr_decay= self._opt.lr / self._opt.nepochs_decay
        new_lr = self._current_lr - lr_decay
        self._update_lr(self._optimizer, self._current_lr, new_lr, 'net')
        self._current_lr = new_lr


    # --- INITIALIZER HELPERS ---

    def _init_create_networks(self):
        # features network
        self._net = NetworksFactory.get_by_name('prob_map_net')
        self._net.init_weights()
        self._net = self._move_net_to_gpu(self._net)

    def _init_train_vars(self):
        # initialize learning rate
        self._current_lr = self._opt.lr

        # initialize optimizers
        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                           lr=self._current_lr)

    def _init_prefetch_inputs(self):
        # prefetch gpu space for images
        self._pos_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.net_image_size, self._opt.net_image_size)
        self._neg_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.net_image_size, self._opt.net_image_size)

        # prefetch gpu space for poses
        self._pos_input_hm = self._Tensor(self._opt.net_image_size, 1, self._opt.net_image_size, self._opt.net_image_size)
        self._gt_neg_hm = Variable(torch.zeros(self._opt.batch_size, 1, self._opt.net_image_size, self._opt.net_image_size)).cuda()

    def _init_losses(self):
        # define loss functions
        self._criterion_pos = torch.nn.MSELoss().cuda()  # mean square error

        # init losses value
        self._loss_pos_hm = Variable(self._Tensor([0]))
        self._loss_neg_hm = Variable(self._Tensor([0]))

    def _init_prefetch_create_hm_vars(self):
        # create hm grid
        images_size = self._opt.net_image_size
        X, Y = np.meshgrid(np.linspace(-1., 1., images_size),  np.linspace(-1., 1., images_size))
        grid = np.stack([Y, X], axis=-1)

        # create hm sigmas
        sigma = np.ones([1, 2]) * self._opt.poses_g_sigma

        # move to gpu, everything related to hm creation is moved to gpu 1 that has more free mem
        self._grid = torch.from_numpy(grid).float().cuda(self._gpu_bb)
        self._sigma = torch.from_numpy(sigma).float().cuda(self._gpu_bb)
        self._threshold_hm = torch.nn.Threshold(0.8, 0)

    # --- FORWARD HELPERS ---

    def _forward(self, keep_data_for_visuals, keep_estimation):
        # get data
        pos_imgs = Variable(self._pos_input_img, volatile=(not self._is_train))
        neg_imgs = Variable(self._neg_input_img, volatile=(not self._is_train))
        gt_hm = Variable(self._pos_input_hm, volatile=(not self._is_train))

        # estim bb and prob
        pos_hm = self._net(pos_imgs)
        neg_hm = self._net(neg_imgs)

        # calculate losses
        self._loss_pos_hm = self._criterion_pos(pos_hm, gt_hm) * self._opt.lambda_bb
        self._loss_neg_hm = self._criterion_pos(neg_hm, self._gt_neg_hm) * self._opt.lambda_prob

        # combined loss (move loss bb to gpu 0)
        total_loss = self._loss_pos_hm + self._loss_neg_hm

        # keep data for visualization
        if keep_data_for_visuals:
            # store visuals
            self._keep_forward_data_for_visualization(pos_imgs, neg_imgs, pos_hm, neg_hm, gt_hm, self._gt_neg_hm)

        # keep estimated data
        if keep_estimation:
            self._keep_estimation(pos_hm, neg_hm)

        return total_loss

    def _keep_forward_data_for_visualization(self, pos_imgs, neg_imgs, pos_hm, neg_hm, gt_pos_hm, gt_neg_hm):
        # store img data
        self._vis_input_pos_img = util.tensor2im(pos_imgs.data)
        self._vis_input_neg_img = util.tensor2im(neg_imgs.data)

        self._vis_gt_pos_hm = gt_pos_hm.cpu().data[0, ...].numpy()
        self._vis_gt_neg_hm = gt_neg_hm.cpu().data[0, ...].numpy()
        self._vis_estim_pos_hm = pos_hm.cpu().data[0, ...].numpy()
        self._vis_estim_neg_hm = neg_hm.cpu().data[0, ...].numpy()

    def _keep_estimation(self, estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob):
        return None

    def _unormalize_bb(self, norm_bb):
        bb = (norm_bb/2 + 0.5) * self._opt.net_image_size
        return bb

    def _gaussian_grid(self, mu, sigma, grid):
        '''
        Generate gaussian grid
        :param mu: must be normalized ...xNx2 (uv)
        :param sigma: must be normalized ...xNx2 (uv)
        :param grid:
        :return:
        '''
        # prepare mu and sigma
        mu = torch.clamp(mu, -1, 1)
        mu = torch.unsqueeze(torch.unsqueeze(mu, -2), -2)
        sigma = torch.unsqueeze(torch.unsqueeze(sigma, -2), -2)

        # generate Gaussians
        z = -torch.sum(torch.pow(grid - mu, 2) / (2 * torch.pow(sigma, 2)), dim=-1)
        G = torch.exp(z)
        G = G / torch.sum(torch.sum(G, dim=-1, keepdim=True), dim=-2, keepdim=True)

        # normalize Gaussians (make every pixel between [0,1])
        G_min = torch.min(torch.min(G, -1, keepdim=True)[0], -2, keepdim=True)[0]
        G_max = torch.max(torch.max(G, -1, keepdim=True)[0], -2, keepdim=True)[0]
        G = (G - G_min) / (G_max - G_min + 1e-8)

        return G