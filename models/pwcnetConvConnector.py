from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra, conv_rep
from flowbias.torchHelpers.ListModule import ListModule


class PWCConvConnector(nn.Module):
    """
    use this class type in a pwcFusion
    """

    def __init__(self, args, connector_kernel_size, conv_layers):
        super(PWCConvConnector, self).__init__()
        self.args = args

        self.num_layers = 5
        # input_sizes = [[9, 15], [17, 30], [34, 60], [68, 120], [135, 240]]
        features = [196, 128, 96, 64, 32]

        self.fusors = nn.ModuleList()
        for i in range(self.num_layers):
            convs = list(conv_rep(features[i], features[i], connector_kernel_size, 1, 1, True, conv_layers - 1).children())
            convs.extend(list(conv_rep(features[i], features[i], connector_kernel_size, 1, 1, False, 1).children()))
            self.fusors.append(nn.Sequential(*convs))
        initialize_msra(self.modules())


class PWCConvAppliedConnector(PWCConvConnector):
    """
    use this class type in a pwcFusion
    """

    def __init__(self, args, connector_kernel_size, conv_layers):
        super(PWCConvAppliedConnector, self).__init__(args, connector_kernel_size, conv_layers)

    def forward(self, x1, l):
        return self.fusors[l](x1)


class PWCConvTrainableConnector(PWCConvConnector):
    """
    use this class type to train the interface weights
    """

    def __init__(self, args, connector_kernel_size, conv_layers):
        super(PWCConvTrainableConnector, self).__init__(args, connector_kernel_size, conv_layers)

    def forward(self, input_dict):
        output_dict = {}
        for i in range(self.num_layers):
            output_dict[f'x1_{i}'] = self.fusors[i](input_dict[f'input_x1_{i}'])
        return output_dict


class PWCConvConnector1(PWCConvAppliedConnector):
    def __init__(self, args, connector_kernel_size):
        super(PWCConvConnector1, self).__init__(args, connector_kernel_size, 1)


class PWCConvConnector2(PWCConvAppliedConnector):
    def __init__(self, args, connector_kernel_size):
        super(PWCConvConnector2, self).__init__(args, connector_kernel_size, 2)


class PWCConvConnector3(PWCConvAppliedConnector):
    def __init__(self, args, connector_kernel_size):
        super(PWCConvConnector3, self).__init__(args, connector_kernel_size, 3)


class PWCTrainableConvConnector11(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector11, self).__init__(args, 1, 1)


class PWCTrainableConvConnector12(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector12, self).__init__(args, 1, 2)


class PWCTrainableConvConnector13(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector13, self).__init__(args, 1, 3)


class PWCTrainableConvConnector31(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector31, self).__init__(args, 3, 1)


class PWCTrainableConvConnector32(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector32, self).__init__(args, 3, 2)


class PWCTrainableConvConnector33(PWCConvTrainableConnector):
    def __init__(self, args):
        super(PWCTrainableConvConnector33, self).__init__(args, 3, 3)


class PWCAppliedConvConnector13(PWCConvAppliedConnector):
    def __init__(self, args):
        super(PWCAppliedConvConnector13, self).__init__(args, 1, 3)


class PWCAppliedConvConnector33(PWCConvAppliedConnector):
    def __init__(self, args):
        super(PWCAppliedConvConnector33, self).__init__(args, 3, 3)


class PWCLinCombAppliedConnector(nn.Module):

    def __init__(self, args):
        super(PWCLinCombAppliedConnector, self).__init__()
        self.args = args

        features = [196, 128, 96, 64, 32]


        self.convs = ListModule(
            *[nn.Conv2d(f, f, kernel_size=1, stride=1, dilation=1, padding=0, bias=False) for f in features])

        self.shifts = [torch.nn.Parameter(torch.zeros((f, f)), requires_grad=True) for f in features]
        for i in range(len(features)):
            self.register_parameter(f"shift_{i}", self.shifts[i])

    def forward(self, x1, l):
        return self.convs[-(l+1)](x1 - self.shifts[-(l+1)][None, :, None, None])