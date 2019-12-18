import sys, os

from utils.eval.model_loading import load_model_parameters, sample_to_torch_batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from datasets.flyingThings3D import FlyingThings3d
from models.pwcnetRecordable import PWCNetRecordable
import numpy as np

model_path = "/visinf/home/vimb01/projects/models/C_PWCNet-onChairs-20191126-113818/checkpoint_best.ckpt"
sample_interface_path = "/data/vimb01/evaluations/C_onThings_interface/"
dataset = FlyingThings3d({},
                         "/data/vimb01/FlyingThings3D_sample401_subset/train/image_clean/left",
                         "/data/vimb01/FlyingThings3D_sample401_subset/train/flow/left", "",
                         photometric_augmentations=False)

layer_id = 0
out_corr_relu_s = {}
x1_s = {}
flow_s = {}
l_s = {}
data_id = 0


def clear_sample():
    global out_corr_relu_s, x1_s, flow_s, l_s, layer_id
    layer_id = 0
    out_corr_relu_s = {}
    x1_s = {}
    flow_s = {}
    l_s = {}


def recorder_func(out_corr_relu, x1, flow, l):
    global out_corr_relu_s, x1_s, flow_s, l_s, layer_id
    ct_str = str(layer_id)

    out_corr_relu_s["out_corr_relu_"+ct_str] = out_corr_relu.cpu().data.numpy()
    x1_s["x1_"+ct_str] = x1.data.cpu().numpy(),
    flow_s["flow_"+ct_str] = flow.data.cpu().numpy(),
    l_s["l_"+ct_str] = np.array(l),
    layer_id += 1


def save_sample():
    global out_corr_relu_s, x1_s, flow_s, l_s, data_id, sample_interface_path

    np.savez(
        sample_interface_path+str(data_id),
        **out_corr_relu_s,
        **x1_s,
        **flow_s,
        **l_s)
    data_id += 1


model = PWCNetRecordable(recorder_func, {})
load_model_parameters(model, model_path, strict=False)
model.eval().cuda()

for ii in range(len(dataset)):
    if ii % 10 == 0:
        print(ii)

    clear_sample()
    model(sample_to_torch_batch(dataset[ii]))
    save_sample()
print("done")
