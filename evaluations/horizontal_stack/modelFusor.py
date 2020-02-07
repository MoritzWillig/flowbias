from flowbias.models import PWCNet, PWCNetFusion, PWCNetLinCombFusion, PWCAppliedConvConnector33, PWCNetWOX1Connection
from flowbias.utils.model_loading import load_model_parameters, save_model

resulting_model_dir = "/data/dataB/fusedModelsWOX1Conn_blind/"

# fusing mode: "blind", "correlation", "trained"
# blind:        no fusing
# correlation:  weighted fusing
# trained:      trained
fusingMode = "blind"

correlation_path = None

connector_class = PWCAppliedConvConnector33

""" base PWCNet fusing
base_model = PWCNet
resulting_blind_model = PWCNet

models = {
    "a": "/data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt",
    "i": "/data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt",
    "h": "/data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt",
    "w": "/data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt"
}

learned_connectors = {
    "ah": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_sintel-20200104-034310/checkpoint_best.ckpt",
    "ai": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AI_33_sintel-20200104-051659/checkpoint_best.ckpt",
    "aw": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AW_33_sintel-20200104-065019/checkpoint_best.ckpt",
    "ha": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HA_33_sintel-20200104-082356/checkpoint_best.ckpt",
    "hi": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HI_33_sintel-20200104-095806/checkpoint_best.ckpt",
    "hw": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HW_33_sintel-20200104-113739/checkpoint_best.ckpt",
    "ia": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IA_33_sintel-20200104-131411/checkpoint_best.ckpt",
    "ih": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IH_33_sintel-20200104-145030/checkpoint_best.ckpt",
    "iw": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IW_33_sintel-20200104-162701/checkpoint_best.ckpt",
    "wa": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WA_33_sintel-20200104-181111/checkpoint_best.ckpt",
    "wh": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WH_33_sintel-20200104-203925/checkpoint_best.ckpt",
    "wi": "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WI_33_sintel-20200104-230204/checkpoint_best.ckpt"
}"""

base_model = PWCNetWOX1Connection
resulting_blind_model = PWCNetWOX1Connection

models = {
    "c": "/data/dataB/models/WOX1_chairs_PWCNetWOX1Connection-20200122-164023/checkpoint_best.ckpt",
    "t": "/data/dataB/models/WOX1_PWCNetWOX1Connection-things-20200127-234143/checkpoint_best.ckpt",
    "s": "/data/dataB/models/WOX1_PWCNetWOX1Connection-sintel-20200127-232828/checkpoint_best.ckpt",
    "k": "/data/dataB/models/WOX1_PWCNetWOX1Connection-kitti-20200128-000101/checkpoint_best.ckpt"
}

learned_connectors = {}


def fuse_and_save_model(
        encoder_path, decoder_path, resulting_model_path, fusing,
        correlation_path=None,
        connector_path=None, connector_class=None):
    encoderModel = base_model({})
    load_model_parameters(encoderModel, encoder_path)

    decoderModel = base_model({})
    load_model_parameters(decoderModel, decoder_path)


    if fusing == "blind":
        resulting_model = resulting_blind_model({})
    elif fusing == "correlation":
        assert(correlation_path is not None)
        connector = PWCNetLinCombFusion({})
        raise RuntimeError("TODO")
        resulting_model = PWCNetFusion(-1, {}, connector)
        resulting_model.connector = connector
    elif fusing == "trained":
        assert (connector_class is not None)
        assert (connector_path is not None)
        connector = connector_class({})
        load_model_parameters(connector, connector_path)
        connector.cuda()
        resulting_model = PWCNetFusion({}, connector)

    # encoder
    resulting_model.feature_pyramid_extractor = encoderModel.feature_pyramid_extractor
    # decoder
    resulting_model.flow_estimators = decoderModel.flow_estimators
    resulting_model.context_networks = decoderModel.context_networks

    resulting_model.cuda()

    save_model(resulting_model, resulting_model_path)


for ename, encoder_model in models.items():
    for dname, decoder_model in models.items():
        if ename is dname:
            continue

        fuse_and_save_model(
            encoder_model, decoder_model, resulting_model_dir+ename+dname+"/", fusingMode,
            connector_path=learned_connectors[ename+dname] if fusingMode != "blind" else None,
            connector_class=connector_class)
print("done")
