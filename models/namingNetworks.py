import torch
import torch.nn as nn

from .pwcnetConvConnector import PWCConvAppliedConnector, PWCLinCombAppliedConnector, PWCAppliedConvConnector13, PWCAppliedConvConnector33
from .pwc_modules import upsample2d_as, initialize_msra, conv_rep
from .pwc_modules import WarpingLayer, FeatureExtractor, ContextNetwork, FlowEstimatorDense
from .correlation_package.correlation import Correlation


class NamingNetwork(nn.Module):
    """
    Modified pwcNet with additional connector between, encoder and decoder, to allow feature mapping
    """

    def __init__(self, args):
        super(NamingNetwork, self).__init__()
        self.args = args
        self.num_chs = [3, 8, 16, 8, 4]
        self.dense = nn.Linear(8*8*4, 4)
        self. = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        _, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw)
        x2_pyramid = self.feature_pyramid_extractor(x2_raw)

        prediction = self.dense(x1_pyramid[0], x2_pyramid[0])

        output_dict = {
            'prediction': prediction
        }
        return output_dict
