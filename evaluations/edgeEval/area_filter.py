import torch
import torch.nn as nn
import torchvision
import math


def get_gaussian_filter(kernel_size, sigma, channels):
    # source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
                          -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
                      )

    # normalization needed ??
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=1,
                                kernel_size=kernel_size, groups=1, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False


class AreaFilter:

    def __init__(self, sigmas, channels=2):
        """
        Uses difference of Gaussians (DoG) to find smooth and noisy areas.
        and -> "Theory of edge detection"
        :param sigmas:
        :param channels:
        """
        self.sigmas = list(sorted(sigmas, reverse=True))

        # the filters capture 2*sigma. +1 to get a symmetric kernel
        self.gaussians = [get_gaussian_filter(sigma * 2 * 2 + 1, sigma, channels) for sigma in sigmas]

        # padding is kernel_size-1/2
        self.paddings = [sigma * 2 for sigma in sigmas]
        self.reflectionPaddings = [nn.ReflectionPad2d(padding) for padding in self.paddings]

    def _dog(self, flow):
        blurred = []
        for gaus, padding, refPad in zip(self.gaussians, self.paddings, self.reflectionPaddings):
            # reflection pad -> gaus -> depad
            blurred.append(padding(refPad(flow))[:, padding:-padding, padding:-padding])

        _, w, h = flow.size()
        n = len(blurred)
        dog = torch.empty((n, w, h),)
        for i, (l_0, l_1) in enumerate(zip(blurred, blurred[1:])):
            dog[i, :, :] = l_1 - l_0
        # dog[i, :, :] = temp - flow #FIXME add this also? -> this contains the remaining noise?
        return dog

    def get_smoothness_score(self, gt_flow, flow):
        dogs = self._dog(gt_flow)

        # how to weight the different levels?
