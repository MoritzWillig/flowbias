from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .pwc_modules import upsample2d_as, initialize_msra, conv
from .pwc_modules import WarpingLayer
from .correlation_package.correlation import Correlation


def get_feature_split(channels, split):
    return int(round(channels*(1-split))), int(round(channels*split))


def applyAndMergeAdd(a, base, expert, weight):
    return base(a) + (weight * expert(a))
    #return torch.addcmul(base(a), weight, expert(a))


class FeatureExtractorExpertAddWOX1(nn.Module):
    def __init__(self, num_chs, num_experts, expert_weight):
        """

        :param num_chs:
        :param num_experts:
        :param expert_split: percentage of features that belong to experts
        """
        super(FeatureExtractorExpertAddWOX1, self).__init__()

        self.num_chs = num_chs
        self._expert_weight = expert_weight
        self.convsBase = nn.ModuleList()
        if expert_weight != 0:
            self.convsExperts = nn.ModuleList([nn.ModuleList() for i in range(num_experts)])

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer_base = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convsBase.append(layer_base)

            if expert_weight != 0:
                for expert_id in range(num_experts):
                    layer_experts = nn.Sequential(
                        conv(ch_in, ch_out, stride=2),
                        conv(ch_out, ch_out)
                    )
                    self.convsExperts[expert_id].append(layer_experts)

    def forward(self, x, expert_id):
        feature_pyramid = []

        if expert_id == -1:
            for conv_base in self.convsBase:
                x = conv_base(x)
                feature_pyramid.append(x)
        else:
            for conv_base, conv_expert in zip(self.convsBase, self.convsExperts[expert_id]):
                # x = conv[1](conv[0](x))
                x_0 = applyAndMergeAdd(x, conv_base[0], conv_expert[0], self._expert_weight)
                x_1 = applyAndMergeAdd(x_0, conv_base[1], conv_expert[1], self._expert_weight)
                feature_pyramid.append(x_1)
                x = x_1
        return feature_pyramid[::-1]


class ContextNetworkExpertAddWOX1(nn.Module):
    """
    The flow prediction from  are not split but added.
    """

    def __init__(self, ch_in, num_experts, expert_weight):
        super(ContextNetworkExpertAddWOX1, self).__init__()
        self._expert_weight = expert_weight

        self.convs_base = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

        if expert_weight != 0:
            convs_experts = []
            for expert_id in range(num_experts):
                convs_experts.append(nn.Sequential(
                    conv(ch_in, 128, 3, 1, 1),
                    conv(128, 128, 3, 1, 2),
                    conv(128, 128, 3, 1, 4),
                    conv(128, 96, 3, 1, 8),
                    conv(96, 64, 3, 1, 16),
                    conv(64, 32, 3, 1, 1),
                    conv(32, 2, isReLU=False)
                ))
            self.convs_experts = nn.ModuleList(convs_experts)

    def forward(self, x, expert_id):
        if expert_id == -1:
            return self.convs_base(x)
        else:
            conv_expert = self.convs_experts[expert_id]
            for i in range(7):
                x = applyAndMergeAdd(x, self.convs_base[i], conv_expert[i], self._expert_weight)
            return x


class FlowEstimatorDenseExpertAddWOX1(nn.Module):

    def __init__(self, ch_in, num_experts, expert_weight, adjust_chs=0):
        super(FlowEstimatorDenseExpertAddWOX1, self).__init__()
        self._expert_weight = expert_weight

        self.conv1_base = conv(ch_in, 128 + adjust_chs)
        self.conv2_base = conv(ch_in + 128 + adjust_chs, 128 + adjust_chs)
        self.conv3_base = conv(ch_in + 256 + adjust_chs, 96 + adjust_chs)
        self.conv4_base = conv(ch_in + 352 + adjust_chs, 64 + adjust_chs)
        self.conv5_base = conv(ch_in + 416 + adjust_chs, 32 + adjust_chs)
        self.conv_last_base = conv(ch_in + adjust_chs + 448, 2, isReLU=False)

        if expert_weight != 0:
            conv1_expert = []
            conv2_expert = []
            conv3_expert = []
            conv4_expert = []
            conv5_expert = []
            conv_last_expert = []
            for expert_id in range(num_experts):
                conv1_expert.append(conv(ch_in, 128 + adjust_chs))
                conv2_expert.append(conv(ch_in + adjust_chs + 128, 128 + adjust_chs))
                conv3_expert.append(conv(ch_in + adjust_chs + 256, 96 + adjust_chs))
                conv4_expert.append(conv(ch_in + adjust_chs + 352, 64 + adjust_chs))
                conv5_expert.append(conv(ch_in + adjust_chs + 416, 32 + adjust_chs))
                conv_last_expert.append(conv(ch_in + adjust_chs + 448, 2, isReLU=False))
            self.conv1_expert = nn.ModuleList(conv1_expert)
            self.conv2_expert = nn.ModuleList(conv2_expert)
            self.conv3_expert = nn.ModuleList(conv3_expert)
            self.conv4_expert = nn.ModuleList(conv4_expert)
            self.conv5_expert = nn.ModuleList(conv5_expert)
            self.conv_last_expert = nn.ModuleList(conv_last_expert)

    def forward(self, x, expert_id):
        if expert_id == -1:
            x1 = torch.cat([self.conv1_base(x), x], dim=1)
            x2 = torch.cat([self.conv2_base(x1), x1], dim=1)
            x3 = torch.cat([self.conv3_base(x2), x2], dim=1)
            x4 = torch.cat([self.conv4_base(x3), x3], dim=1)
            x5 = torch.cat([self.conv5_base(x4), x4], dim=1)
            x_out = self.conv_last_base(x5)
            return x5, x_out
        else:
            x1 = torch.cat([applyAndMergeAdd(x, self.conv1_base, self.conv1_expert[expert_id], self._expert_weight), x], dim=1)
            x2 = torch.cat([applyAndMergeAdd(x1, self.conv2_base, self.conv2_expert[expert_id], self._expert_weight), x1], dim=1)
            x3 = torch.cat([applyAndMergeAdd(x2, self.conv3_base, self.conv3_expert[expert_id], self._expert_weight), x2], dim=1)
            x4 = torch.cat([applyAndMergeAdd(x3, self.conv4_base, self.conv4_expert[expert_id], self._expert_weight), x3], dim=1)
            x5 = torch.cat([applyAndMergeAdd(x4, self.conv5_base, self.conv5_expert[expert_id], self._expert_weight), x4], dim=1)
            x_out = applyAndMergeAdd(x5, self.conv_last_base, self.conv_last_expert[expert_id], self._expert_weight)
            return x5, x_out


class PWCExpertAddNetWOX1(nn.Module):
    def __init__(
            self, args, num_experts, expert_weight, div_flow=0.05, adjust_decover_conv_layers=True,
            has_encoder_experts=True, has_decoder_experts=True,
            has_context_experts=None, ignore_missing_experts=False,
            split_to_gpus=None):
        super(PWCExpertAddNetWOX1, self).__init__()
        self.args = args
        self._num_experts = num_experts
        self._expert_weight = expert_weight
        self._div_flow = div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self._cuda_self_configurated = False

        if args.cuda:
            if split_to_gpus is not None:
                if len(split_to_gpus) == 1:
                    gpu_split0 = torch.cuda.device(split_to_gpus[0])
                    gpu_split1 = torch.cuda.device(split_to_gpus[0])
                elif len(split_to_gpus) == 2:
                    gpu_split0 = torch.cuda.device(split_to_gpus[0])
                    gpu_split1 = torch.cuda.device(split_to_gpus[1])
                else:
                    raise ValueError("model splits other than {0,1} are not implemented")
                self._cuda_self_configurated = True
        else:
            if split_to_gpus is not None:
                raise ValueError("passed split_to_gpus parameter, but cuda was False")
            gpu_split0 = None
            gpu_split1 = None

        if has_context_experts is None:
            has_context_experts = has_decoder_experts

        self.has_encoder_experts = has_encoder_experts
        self.has_decoder_experts = has_decoder_experts
        self.has_context_experts = has_context_experts
        self.ignore_missing_experts = ignore_missing_experts

        feature_extractor_weight = self._expert_weight if has_encoder_experts else 0
        flow_estimator_weight = self._expert_weight if has_decoder_experts else 0
        context_network_weight = self._expert_weight if has_context_experts else 0

        self.feature_pyramid_extractor = FeatureExtractorExpertAddWOX1(self.num_chs, num_experts, feature_extractor_weight).to(gpu_split0)
        self.warping_layer = WarpingLayer().to(gpu_split0)

        self.flow_estimators = nn.ModuleList().to(gpu_split1)
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr
                ch = 0
            else:
                num_ch_in = self.dim_corr + 2

            layer = FlowEstimatorDenseExpertAddWOX1(num_ch_in, num_experts, flow_estimator_weight, 0 if adjust_decover_conv_layers else ch).to(gpu_split1)
            self.flow_estimators.append(layer)

        self.context_networks = ContextNetworkExpertAddWOX1(
            self.dim_corr + 32 + 2 + 448 + 2 - self.num_chs[-(self.output_level+1)], num_experts, context_network_weight).to(gpu_split1)

        initialize_msra(self.modules())

    def self_cuda_from_args(self):
        """
        indicates if the modules are already moved to the gpus
        :return:
        """
        return self._cuda_self_configurated

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

        encoder_expert_id = encoder_expert_id if (not self.ignore_missing_experts) or self.has_encoder_experts else -1
        decoder_expert_id = decoder_expert_id if (not self.ignore_missing_experts) or self.has_decoder_experts else -1
        context_expert_id = context_expert_id if (not self.ignore_missing_experts) or self.has_context_experts else -1

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

            # move tensors if the model split between gpus
            out_corr_relu = out_corr_relu.to(self.flow_estimators[l].device(), non_blocking=True)
            flow = flow.to(self.flow_estimators[l].device(), non_blocking=True)

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


class CTSKPWCExpertNetWOX1Add01(PWCExpertAddNetWOX1):

    def __init__(self, args, div_flow=0.05):
        super().__init__(args, 4, 0.1, div_flow=div_flow)


class CTSPWCExpertNetWOX1Add01(PWCExpertAddNetWOX1):

    def __init__(self, args, div_flow=0.05):
        super().__init__(args, 3, 0.1, div_flow=div_flow)


class CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly(PWCExpertAddNetWOX1):

    def __init__(self, args, div_flow=0.05, ignore_missing_experts=False):
        super().__init__(args, 4, 0.1, div_flow=div_flow, has_decoder_experts=False,
                         ignore_missing_experts=ignore_missing_experts)


class CTSKPWCExpertNetWOX1Add01DecoderExpertsOnly(PWCExpertAddNetWOX1):

    def __init__(self, args, div_flow=0.05, ignore_missing_experts=False):
        super().__init__(args, 4, 0.1, div_flow=div_flow, has_encoder_experts=False,
                         ignore_missing_experts=ignore_missing_experts)
