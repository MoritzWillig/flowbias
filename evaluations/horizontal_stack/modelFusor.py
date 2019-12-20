#from __future__ import absolute_import, division, print_function

import torch

from models import PWCNet, PWCNetFusion, PWCConvAppliedConnector, PWCNetLinCombFusion, PWCNetConvFusion
from utils.eval.model_loading import load_model_parameters, save_model

encoder_path = "/visinf/home/vimb01/projects/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt"
decoder_path = "/visinf/home/vimb01/projects/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt"

resulting_model_path = "/visinf/home/vimb01/projects/fusedModels/A_I_blind/"

# fusing type: "blind", "correlation", "trained"
# blind:        no fusing
# correlation:  weighted fusing
# trained:      trained
fusing = "blind"

correlation_path = None

connector_path = None
connector_kernel_size = 1


encoderModel = PWCNet({})
load_model_parameters(encoderModel, encoder_path)

decoderModel = PWCNet({})
load_model_parameters(decoderModel, decoder_path)


if fusing == "blind":
    resultingModel = PWCNet({})
elif fusing == "correlation":
    assert(correlation_path is not None)
    connector = PWCNetLinCombFusion({})
    raise RuntimeError("TODO")
    resultingModel = PWCNetFusion(-1, {}, connector)
    resultingModel.connector = connector
elif fusing == "trained":
    assert (connector_path is not None)
    connector = PWCNetConvFusion(connector_kernel_size, {})
    connector.load_state_dict(torch.load(connector_path))
    resultingModel = PWCNetFusion(-1, {}, connector)
    resultingModel.connector = connector


# encoder
resulting_model = encoderModel.feature_pyramid_extractor
# decoder
resulting_model.flow_estimators = decoderModel.flow_estimators
resulting_model.context_networks = decoderModel.context_networks

save_model(resulting_model, resulting_model_path)


