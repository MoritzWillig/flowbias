import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numbers


#taken from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, * kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def round_up_to_odd(f):
    return math.ceil(f) // 2 * 2 + 1


class AreaFilter:

    def find_kernel_size(self, sigma):
        # include 2 sigma and round to next odd number
        return round_up_to_odd(sigma)

    def __init__(self, sigmas, channels=2):
        """
        Uses difference of Gaussians (DoG) to find smooth and noisy areas.
        and -> "Theory of edge detection"
        :param sigmas:
        :param channels:
        """
        self.sigmas = list(sorted(sigmas, reverse=True))

        # the filters capture 2*sigma. +1 to get a symmetric kernel
        kernel_sizes = [self.find_kernel_size(sigma) for sigma in sigmas]
        self.gaussians = [GaussianSmoothing(channels, kernel_size, sigma).cuda() for sigma, kernel_size in zip(sigmas, kernel_sizes)]

        # padding is kernel_size-1/2
        self.paddings = [(kernel_size//2) for kernel_size in kernel_sizes]
        self._max_pad = self.paddings[-1]

    def compute_dog(self, flow):
        max_pad = self._max_pad
        blurred = []
        for gaus, padding in zip(self.gaussians, self.paddings):
            # padded gaus -> depad
            g = gaus(flow)
            g = torch.sum(torch.abs(g), dim=1)
            pad = max_pad - padding
            p = g if pad == 0 else g[:, pad:-pad, pad:-pad]
            blurred.append(p)
            #print("->",p.size())

        b, _, w, h = flow.size()
        n = len(blurred)
        dog = torch.empty((b, n, w-2*max_pad, h-2*max_pad))
        for i, (l_0, l_1) in enumerate(zip(blurred, blurred[1:])):
            dog[:, i, :, :] = l_1 - l_0
        return dog

    def get_smoothness_score(self, gt_flow, flow):
        dogs = self.compute_dog(gt_flow)

        # how to weight the different levels?
