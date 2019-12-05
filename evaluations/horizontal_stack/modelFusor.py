from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from models import PWCNet, PWCNetFusion, PWCConvAppliedConnector, PWCNetLinCombFusion, PWCNetConvFusion

encoder_path = "/visinf/home/vimb01/projects/models/A_chairs_PWCNet-20191121-171532/checkpoint_best.ckpt"
decoder_path = "/visinf/home/vimb01/projects/models/C_chairs_PWCNet-20191126-113818/checkpoint_best.ckpt"


correlation_path = ""

connector_path = ""
connector_kernel_size = 1

resultingModelPath = ""


encoderModel = PWCNet({})
encoderModel.load_state_dict(torch.load(encoder_path))

decoderModel = PWCNet({})
decoderModel.load_state_dict(torch.load(decoder_path))

if correlation_path is not None:
    connector = PWCNetLinCombFusion({})
else:
    connector = PWCNetConvFusion(connector_kernel_size, {})
connector.load_state_dict(torch.load(connector_path))


resultingModel = PWCNetFusion(-1, {}, connector)
# encoder
resultingModel.feature_pyramid_extractor = encoderModel.feature_pyramid_extractor
# feature mapping
resultingModel.connector = connector
# decoder
resultingModel.flow_estimators = decoderModel.flow_estimators
resultingModel.context_networks = decoderModel.context_networks


torch.save(resultingModel.state_dict(), resultingModelPath)
