#from __future__ import absolute_import, division, print_function

import torch

from flowbias.models import PWCNet, PWCNetFusion, PWCConvAppliedConnector, PWCNetLinCombFusion, PWCNetConvFusion
from flowbias.utils.eval.model_loading import load_model_parameters, save_model

encoder_path = "/visinf/home/vimb01/projects/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt"
decoder_path = "/visinf/home/vimb01/projects/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt"

resulting_model_path = "/visinf/home/vimb01/projects/fusedModels/I_H_blind/"

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
    resulting_model = PWCNet({})
elif fusing == "correlation":
    assert(correlation_path is not None)
    connector = PWCNetLinCombFusion({})
    raise RuntimeError("TODO")
    resulting_model = PWCNetFusion(-1, {}, connector)
    resulting_model.connector = connector
elif fusing == "trained":
    assert (connector_path is not None)
    connector = PWCNetConvFusion(connector_kernel_size, {})
    connector.load_state_dict(torch.load(connector_path))
    resulting_model = PWCNetFusion(-1, {}, connector)
    resulting_model.connector = connector


# encoder
resulting_model.feature_pyramid_extractor = encoderModel.feature_pyramid_extractor
# decoder
resulting_model.flow_estimators = decoderModel.flow_estimators
resulting_model.context_networks = decoderModel.context_networks

resulting_model.cuda()

save_model(resulting_model, resulting_model_path)


