import torch
import numpy as np

class JointsUtils:
    def __init__(self, num_sigmas, size, sigma, device_master, max=3, min=0.0001):
        X, Y = np.meshgrid(np.linspace(-1., 1., size), np.linspace(-1., 1., size))
        grid = np.stack([Y, X], axis=-1)
        sigma = np.ones([num_sigmas, 2]) * sigma

        self._max = max
        self._min = min
        self._size = size
        self._grid = torch.from_numpy(grid).float().to(device_master)
        sigma = torch.from_numpy(sigma).float().to(device_master)
        self._sigma = torch.Tensor.unsqueeze(torch.Tensor.unsqueeze(sigma, -2), -2)


    def gaussian_grid(self, mu):
        '''
        Generate gaussian grid
        :param mu:...xNx2 (ij)
        :return:
        '''
        # prepare mu and sigma
        # mu = self._norm_mu(mu, self._size)
        mu = torch.Tensor.clamp(mu, -1, 1)
        mu = torch.Tensor.unsqueeze(torch.Tensor.unsqueeze(mu, -2), -2)

        # generate Gaussians
        z = -torch.Tensor.sum(torch.Tensor.pow(self._grid - mu, 2) / (2 * torch.Tensor.pow(self._sigma, 2)), dim=-1)
        G = torch.Tensor.exp(z)
        G = G / (torch.Tensor.sum(torch.Tensor.sum(G, dim=-1, keepdim=True), dim=-2, keepdim=True) + 1e-8)
        G = self._norm_hm(G)
        return G

    def _norm_mu(self, pose, image_size):
        return (pose / (image_size-1) - 0.5) * 2

    def _norm_hm(self, hm):
        hm_min = torch.Tensor.min(torch.Tensor.min(hm, -1, keepdim=True)[0], -2, keepdim=True)[0]
        hm_max = torch.Tensor.max(torch.Tensor.max(hm, -1, keepdim=True)[0], -2, keepdim=True)[0]
        return (hm - hm_min) / (hm_max - hm_min + 1e-8) * (self._max-self._min) + self._min

    def get_max_pixel_activation(self, hm):
        s = hm.size(-1)
        val, index = torch.Tensor.max(hm.view(1, -1), 1)
        return index % s, index / s, val
