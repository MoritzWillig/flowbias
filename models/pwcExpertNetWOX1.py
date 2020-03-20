from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra, conv
from .pwc_modules import WarpingLayer
from .correlation_package.correlation import Correlation


def get_feature_split(channels, split):
    return int(round(channels*(1-split))), int(round(channels*split))


def applyAndMerge(a, base, expert):
    return torch.cat([base(a), expert(a)], dim=1)


class FeatureExtractorExpertWOX1(nn.Module):
    def __init__(self, num_chs, num_experts, expert_split):
        """

        :param num_chs:
        :param num_experts:
        :param expert_split: percentage of features that belong to experts
        """
        super(FeatureExtractorExpertWOX1, self).__init__()

        self.num_chs = num_chs
        self.convsBase = nn.ModuleList()
        self.convsExperts = nn.ModuleList([nn.ModuleList() for i in range(num_experts)])

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            ch_out_base, ch_out_experts = get_feature_split(ch_out, expert_split)

            layer_base = nn.Sequential(
                conv(ch_in, ch_out_base, stride=2),
                conv(ch_out, ch_out_base)
            )
            self.convsBase.append(layer_base)

            for expert_id in range(num_experts):
                layer_experts = nn.Sequential(
                    conv(ch_in, ch_out_experts, stride=2),
                    conv(ch_out, ch_out_experts)
                )
                self.convsExperts[expert_id].append(layer_experts)

    def forward(self, x, expert_id):
        feature_pyramid = []
        for conv_base, conv_expert in zip(self.convsBase, self.convsExperts[expert_id]):
            # x = conv[1](conv[0](x))
            x_0 = applyAndMerge(x, conv_base[0], conv_expert[0])
            x_1 = applyAndMerge(x_0, conv_base[1], conv_expert[1])
            feature_pyramid.append(x_1)
            x = x_1
        return feature_pyramid[::-1]


class ContextNetworkExpertWOX1(nn.Module):
    """
    The flow prediction from  are not split but added.
    """

    def __init__(self, ch_in, num_experts, expert_split):
        super(ContextNetworkExpertWOX1, self).__init__()

        b_128, e_128 = get_feature_split(128, expert_split)
        b_96, e_96 = get_feature_split(96, expert_split)
        b_64, e_64 = get_feature_split(64, expert_split)
        b_32, e_32 = get_feature_split(32, expert_split)

        self.convs_base = nn.Sequential(
            conv(ch_in, b_128, 3, 1, 1),
            conv(128, b_128, 3, 1, 2),
            conv(128, b_128, 3, 1, 4),
            conv(128, b_96, 3, 1, 8),
            conv(96, b_64, 3, 1, 16),
            conv(64, b_32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

        convs_experts = []
        for expert_id in range(num_experts):
            convs_experts.append(nn.Sequential(
                conv(ch_in, e_128, 3, 1, 1),
                conv(128, e_128, 3, 1, 2),
                conv(128, e_128, 3, 1, 4),
                conv(128, e_96, 3, 1, 8),
                conv(96, e_64, 3, 1, 16),
                conv(64, e_32, 3, 1, 1),
                conv(32, 2, isReLU=False)
            ))
        self.convs_experts = nn.ModuleList(convs_experts)

    def forward(self, x, expert_id):
        conv_expert = self.convs_experts[expert_id]
        for i in range(6):
            x = applyAndMerge(x, self.convs_base[i], conv_expert[i])

        # the resulting features cant be split. Instead we add them together.
        flow_base = self.convs_base[6](x)
        flow_expert = conv_expert[6](x)
        return flow_base + flow_expert


class FlowEstimatorDenseExpertWOX1(nn.Module):

    def __init__(self, ch_in, num_experts, expert_split):
        super(FlowEstimatorDenseExpertWOX1, self).__init__()

        b_128, e_128 = get_feature_split(128, expert_split)
        b_96, e_96 = get_feature_split(96, expert_split)
        b_64, e_64 = get_feature_split(64, expert_split)
        b_32, e_32 = get_feature_split(32, expert_split)

        self.conv1_base = conv(ch_in, b_128)
        self.conv2_base = conv(ch_in + 128, b_128)
        self.conv3_base = conv(ch_in + 256, b_96)
        self.conv4_base = conv(ch_in + 352, b_64)
        self.conv5_base = conv(ch_in + 416, b_32)
        self.conv_last_base = conv(ch_in + 448, 2, isReLU=False)

        conv1_expert = []
        conv2_expert = []
        conv3_expert = []
        conv4_expert = []
        conv5_expert = []
        conv_last_expert = []
        for expert_id in range(num_experts):
            conv1_expert.append(conv(ch_in, e_128))
            conv2_expert.append(conv(ch_in + 128, e_128))
            conv3_expert.append(conv(ch_in + 256, e_96))
            conv4_expert.append(conv(ch_in + 352, e_64))
            conv5_expert.append(conv(ch_in + 416, e_32))
            conv_last_expert.append(conv(ch_in + 448, 2, isReLU=False))
        self.conv1_expert = nn.ModuleList(conv1_expert)
        self.conv2_expert = nn.ModuleList(conv2_expert)
        self.conv3_expert = nn.ModuleList(conv3_expert)
        self.conv4_expert = nn.ModuleList(conv4_expert)
        self.conv5_expert = nn.ModuleList(conv5_expert)
        self.conv_last_expert = nn.ModuleList(conv_last_expert)

    def forward(self, x, expert_id):
        x1 = torch.cat([applyAndMerge(x, self.conv1_base, self.conv1_expert[expert_id]), x], dim=1)
        x2 = torch.cat([applyAndMerge(x1, self.conv2_base, self.conv2_expert[expert_id]), x1], dim=1)
        x3 = torch.cat([applyAndMerge(x2, self.conv3_base, self.conv3_expert[expert_id]), x2], dim=1)
        x4 = torch.cat([applyAndMerge(x3, self.conv4_base, self.conv4_expert[expert_id]), x3], dim=1)
        x5 = torch.cat([applyAndMerge(x4, self.conv5_base, self.conv5_expert[expert_id]), x4], dim=1)

        # flow prediction features can't be split. So we add them instead.
        x_out_base = self.conv_last_base(x5)
        x_out_expert = self.conv_last_expert[expert_id](x5)
        x_out = x_out_base + x_out_expert
        return x5, x_out


class PWCExpertNetWOX1(nn.Module):
    def __init__(self, args, num_experts, expert_split, div_flow=0.05):
        super(PWCExpertNetWOX1, self).__init__()
        self.args = args
        self._num_experts = num_experts
        self._expert_split = expert_split
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractorExpertWOX1(self.num_chs, num_experts, expert_split)
        self.warping_layer = WarpingLayer()

        self.flow_estimators = nn.ModuleList()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr
                ch = 0
            else:
                num_ch_in = self.dim_corr + 2

            layer = FlowEstimatorDenseExpertWOX1(num_ch_in, num_experts, expert_split)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetworkExpertWOX1(
            self.dim_corr + 32 + 2 + 448 + 2 - self.num_chs[-(self.output_level+1)], num_experts, expert_split)

        initialize_msra(self.modules())

    def forward(self, input_dict):
        # assuming each sample in the batch is from the same dataset
        if 'dataset' in input_dict:
            expert_id = input_dict['dataset'] if isinstance(input_dict['dataset'], int) else input_dict['dataset'][0]
            encoder_expert_id = expert_id
            decoder_expert_id = expert_id
            context_expert_id = expert_id
        else:
            encoder_expert_id = input_dict['encoder_expert_id'] if isinstance(input_dict['encoder_expert_id'], int) else input_dict['encoder_expert_id'][0]
            decoder_expert_id = input_dict['decoder_expert_id'] if isinstance(input_dict['decoder_expert_id'], int) else input_dict['decoder_expert_id'][0]
            context_expert_id = input_dict['context_expert_id'] if isinstance(input_dict['context_expert_id'], int) else input_dict['context_expert_id'][0]
        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw, encoder_expert_id) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw, encoder_expert_id) + [x2_raw]

        # outputs
        output_dict = {}
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)

            # correlation
            out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # flow estimator
            if l == 0:
                x_intm, flow = self.flow_estimators[l](out_corr_relu, decoder_expert_id)
            else:
                x_intm, flow = self.flow_estimators[l](torch.cat([out_corr_relu, flow], dim=1), decoder_expert_id)
            # The name 'x_intm' is left unchanged for consistence with the original architecture. However, it now only
            # depends on the correlation and the predicted flow from the upper layers.

            # upsampling or post-processing
            if l != self.output_level:
                flows.append(flow)
            else:
                flow_res = self.context_networks(torch.cat([x_intm, flow], dim=1), context_expert_id)
                flow = flow + flow_res
                flows.append(flow)                
                break

        output_dict['flow'] = flows

        if self.training:
            return output_dict
        else:
            output_dict_eval = {}
            out_flow = upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
            output_dict_eval['flow'] = out_flow
            return output_dict_eval


class CTSKPWCExpertNet02WOX1(PWCExpertNetWOX1):
    def __init__(self, args, div_flow=0.05):
        num_experts = 4
        expert_split = 0.2
        super().__init__(args, num_experts, expert_split, div_flow)


class CTSPWCExpertNet02WOX1(PWCExpertNetWOX1):
    def __init__(self, args, div_flow=0.05):
        num_experts = 3
        expert_split = 0.2
        super().__init__(args, num_experts, expert_split, div_flow=div_flow)
