import torch
from collections import OrderedDict
import util.util as util
import util.plots as plots
from util.joints_utils import JointsUtils
from .models import BaseModel
from networks.networks import NetworksFactory
import numpy as np

class ObjectDetectorNetModel(BaseModel):
    def __init__(self, opt):
        super(ObjectDetectorNetModel, self).__init__(opt)
        self._name = 'ObjectDetectorNetModel'
        self._gpu_bb = 1 if len(self._gpu_ids) > 1 else 0  # everything related bb moved to gpu 1 that has more free mem

        # create networks
        self._init_create_networks()
        self._init_prefetch_create_hm_vars()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # init losses
        if self._is_train:
            self._init_losses()

    def set_input(self, input):
        # copy images efficiently
        self._pos_input_img = input['pos_img'].to(self._device)
        self._neg_input_img = input['neg_img'].to(self._device)
        self._input_hms = self._pose_to_hm(torch.unsqueeze(input['pos_norm_pose'], 1).to(self._device))  # ij coords

        # store paths
        self._pos_input_img_path = input['pos_img_path']
        self._neg_input_img_path = input['neg_img_path']

    def _pose_to_hm(self, joints):
        with torch.no_grad():
            hm = self._joint_utils.gaussian_grid(joints)
            return hm

    def forward(self, keep_data_for_visuals=False, keep_estimation=False):
        # clean previous gradients
        if self._is_train:
           self._optimizer.zero_grad()

        # forward pass
        loss_pose = self._forward(keep_data_for_visuals, keep_estimation)

        # compute new gradients
        if self._is_train:
            loss_pose.backward()

    def test(self, image, do_normalize_output=False):
        with torch.no_grad():
            estim_hm = self._net(image.to(self._device))
            u_max, v_max, val = self._joint_utils.get_max_pixel_activation(estim_hm)

        return estim_hm.detach().cpu().numpy(), (u_max, v_max), val

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
                            ('neg_img', self._neg_input_img_path)
                            ])

    def get_current_errors(self):
        return OrderedDict([('pos_p', self._loss_pos_p.item()),
                            ('neg_p', self._loss_neg_p.item())
        ])

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_last_saved_estimation(self):
        """
        Returns last model estimation with flag keep_estimation=True
        """
        return None

    def get_last_saved_visuals(self):
        """
        Returns last model visuals with flag keep_data_for_visuals=True
        """
        # visuals return dictionary
        visuals = OrderedDict()
        visuals['input_pos'] = self._vis_input_pos_img
        visuals['input_neg'] = self._vis_input_neg_img

        visuals['estim_pos_p'] = self._vis_pos_p
        visuals['input_pos_p'] = self._vis_input_pos_p

        visuals['estim_neg_p'] = self._vis_neg_p
        visuals['input_neg_p'] = self._vis_input_neg_p

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

    def update_learning_rate(self, epoch):
        every = 100
        factor = 0.5
        new_lr = self._current_lr * factor if (epoch + 1) % every == 0 else self._current_lr
        if new_lr != self._current_lr:
            self._update_lr(self._optimizer, self._current_lr, new_lr, 'net')
            self._current_lr = new_lr


    # --- INITIALIZER HELPERS ---

    def _init_create_networks(self):
        # features network
        self._net = NetworksFactory.get_by_name('unet_small')
        self._net = self._move_net_to_gpu(self._net)

    def _init_prefetch_create_hm_vars(self):
        sigma = 0.05
        self._joint_utils = JointsUtils(1, self._opt.net_image_size, sigma, self._device)
        self._zero_hms = torch.ones([self._opt.batch_size, 1, self._opt.net_image_size, self._opt.net_image_size]).to(self._device) * 0.0001

    def _init_train_vars(self):
        # initialize learning rate
        self._current_lr = self._opt.lr

        # initialize optimizers
        # self._optimizer = torch.optim.SGD(self._net.parameters(),
        #                                   lr=self._current_lr, weight_decay=5e-4, momentum=0.9)
        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                           lr=self._current_lr, weight_decay=5e-4)

    def _init_losses(self):
        # define loss functions
        self._criterion_pos = torch.nn.MSELoss().to(self._device)

        # init losses value
        self._loss_pos_p = torch.zeros(1, requires_grad=True).to(self._device)

    def _forward(self, keep_data_for_visuals, keep_estimation):
        with torch.set_grad_enabled(self._is_train):
            # estim bb and prob
            import time
            s = time.time()
            pos_p = self._net(self._pos_input_img)
            print(time.time() - s)
            neg_p = self._net(self._neg_input_img)

            # calculate losses
            self._loss_pos_p = self._criterion_pos(pos_p, self._input_hms) * self._opt.lambda_bb * 10
            self._loss_neg_p = self._criterion_pos(neg_p, self._zero_hms) * self._opt.lambda_bb

            # combined loss (move loss bb to gpu 0)
            total_loss = self._loss_pos_p + self._loss_neg_p

            # keep data for visualization
            if keep_data_for_visuals:
                # store visuals
                self._keep_forward_data_for_visualization(self._pos_input_img, self._neg_input_img, pos_p, neg_p,
                                                          self._input_hms, self._zero_hms)

            # keep estimated data
            if keep_estimation:
                pass

        return total_loss

    def _keep_forward_data_for_visualization(self, pos_imgs, neg_imgs, pos_p, neg_p, gt_pos_p, gt_neg_p):
        # store img data
        self._vis_input_pos_img = util.tensor2im(pos_imgs.clone().detach(), scale=1)
        self._vis_input_neg_img = util.tensor2im(neg_imgs.clone().detach(), scale=1)

        pos_img_hm = util.tensor2im(pos_imgs[0, ...].clone().detach().cpu(), to_numpy=True, scale=1)
        print(torch.max(pos_p[0, 0, ...]), torch.min(pos_p[0, 0, ...]))
        self._vis_pos_p = plots.plot_overlay_attention(pos_img_hm, pos_p[0, 0, ...].clone().detach().cpu())
        self._vis_input_pos_p = plots.plot_overlay_attention(pos_img_hm, gt_pos_p[0, 0, ...].clone().detach().cpu())

        neg_img_hm = util.tensor2im(neg_imgs[0, ...].clone().detach().cpu(), to_numpy=True, scale=1)
        self._vis_neg_p = plots.plot_overlay_attention(neg_img_hm, neg_p[0, 0, ...].clone().detach().cpu())
        self._vis_input_neg_p = plots.plot_overlay_attention(neg_img_hm, gt_neg_p[0, 0, ...].clone().detach().cpu())

    def _keep_estimation(self, estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob):
        return None

    def _unormalize_pose(self, norm_pose):
        return (norm_pose/2.0 + 0.5) * self._opt.net_image_size