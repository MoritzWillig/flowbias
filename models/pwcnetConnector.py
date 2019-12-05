from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra, conv_rep
from torchHelpers.ListModule import ListModule

class PWCConvConnector(nn.Module):

    def __init__(self, connector_kernel_size, args):
        super(PWCConvConnector, self).__init__()
        self.args = args

        self.num_layers = 5
        input_sizes = [[9, 15], [17, 30], [34, 60], [68, 120], [135, 240]]
        features = [196, 128, 96, 64, 32]

        connector_layers = 1

        self.fusors = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusors.append(conv_rep(features[i], features[i], connector_kernel_size, 1, 1, True, connector_layers))
        initialize_msra(self.modules())

    def forward(self, input_dict):
        output_dict = {}
        for i in range(self.num_layers):
            output_dict[f'target_x1_{i}'] = self.fusors[i](input_dict[f'input_x1_{i}'])
        return output_dict


class PWCConvAppliedConnector(nn.Module):

    def __init__(self, connector_kernel_size, args):
        super(PWCConvAppliedConnector, self).__init__()
        self.args = args

        self.num_layers = 5
        # input_sizes = [[9, 15], [17, 30], [34, 60], [68, 120], [135, 240]]
        features = [196, 128, 96, 64, 32]

        connector_layers = 1

        self.fusors = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusors.append(conv_rep(features[i], features[i], connector_kernel_size, 1, 1, True, connector_layers))
        initialize_msra(self.modules())

    def forward(self, x1, l):
        return self.fusors[-(l+1)](x1)


class PWCConvConnector1(PWCConvAppliedConnector):
    def __init__(self, args):
        super(PWCConvConnector1, self).__init__(1, args)


class PWCConvConnector3(PWCConvAppliedConnector):
    def __init__(self, args):
        super(PWCConvConnector3, self).__init__(1, args)


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