from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra, conv
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork
from .correlation_package.correlation import Correlation


class SecFlowEstimatorDense(nn.Module):
    def __init__(self, ch_in, adjust_chs=0):
        super(SecFlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128+adjust_chs)
        self.conv2 = conv(ch_in + adjust_chs + 128, 128+adjust_chs)
        self.conv3 = conv(ch_in + adjust_chs + 256, 96+adjust_chs)
        self.conv4 = conv(ch_in + adjust_chs + 352, 64+adjust_chs)
        self.conv5 = conv(ch_in + adjust_chs + 416, 32+adjust_chs)
        self.conv_last = conv(ch_in + adjust_chs + 448, 2, isReLU=False)
        self.conv_last_sec = conv(ch_in + adjust_chs + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        x_out_sec = self.conv_last_sec(x5)
        return x5, x_out, x_out_sec


class PWCNetWOX1SecondaryFlow(nn.Module):
    
    def __init__(self, args, div_flow=0.05):
        super(PWCNetWOX1SecondaryFlow, self).__init__()
        self.args = args
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.flow_estimators = nn.ModuleList()
        self.dim_corr = ((self.search_range * 2 + 1) ** 2)
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + self.dim_corr
            else:
                num_ch_in = self.dim_corr + self.dim_corr + 2 + 2

            layer = SecFlowEstimatorDense(num_ch_in)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetwork(self.dim_corr + self.dim_corr + 2 + 2 + 448 + 2 + 2)

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        flows = []
        sec_flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow_sec = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x2_warp_sec = x2
            else:
                flow = upsample2d_as(flow, x1, mode="bilinear")
                flow_sec = upsample2d_as(flow_sec, x1, mode="bilinear")
                x2_warp = self.warping_layer(x2, flow, height_im, width_im, self._div_flow)
                x2_warp_sec = self.warping_layer(x2, flow_sec, height_im, width_im, self._div_flow)

            # correlation
            out_corr = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
            out_corr_sec = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp_sec)
            out_corr_relu = self.leakyRELU(out_corr)
            out_corr_sec_relu = self.leakyRELU(out_corr_sec)

            # flow estimator
            if l == 0:
                x_intm, flow, flow_sec = self.flow_estimators[l](torch.cat([out_corr_relu, out_corr_sec_relu], dim=1))
            else:
                x_intm, flow, flow_sec = self.flow_estimators[l](torch.cat([out_corr_relu, out_corr_sec_relu, flow, flow_sec], dim=1))

            # upsampling or post-processing
            if l != self.output_level:
                flows.append(flow)
                sec_flows.append(flow_sec)
            else:
                flow_res = self.context_networks(torch.cat([x_intm, flow, flow_sec], dim=1))
                flow = flow + flow_res
                flows.append(flow)
                sec_flows.append(flow_sec)
                break

        output_dict['flow'] = flows
        output_dict['sec_flow'] = sec_flows

        if self.training:
            return output_dict
        else:
            output_dict_eval = {}
            out_flow = upsample2d_as(flow, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
            output_dict_eval['flow'] = out_flow
            return output_dict_eval
