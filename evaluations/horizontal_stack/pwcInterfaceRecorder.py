from pathlib import Path
import numpy as np

from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.datasets.flyingThings3D import FlyingThings3d
from flowbias.datasets.flyingchairs import FlyingChairsFull
from flowbias.datasets.subsampledDataset import SubsampledDataset
from flowbias.datasets.kitti_combined import KittiComb2015Train
from flowbias.models.pwcnetRecordable import PWCNetRecordable

models = {
    "A": "/data/dataB/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt",
    "I": "/data/dataB/models/I_PWCNet-things_20191209-131019/checkpoint_best.ckpt",
    "H": "/data/dataB/models/H_PWCNet-sintel-20191209-150448/checkpoint_best.ckpt",
    "W": "/data/dataB/models/W_PWCNet-kitti-20191216-124247/checkpoint_best.ckpt"
}

datasets = {
    "chairs": FlyingChairsFull({}, "/data/dataB/datasets/FlyingChairs_sample402/data/", photometric_augmentations=False),
    "things": FlyingThings3d({},
                         "/data/dataB/datasets/FlyingThings3D_sample401_subset/train/image_clean/left",
                         "/data/dataB/datasets/FlyingThings3D_sample401_subset/train/flow/left", "",
                         photometric_augmentations=False),
    "sintel": SubsampledDataset({}, "/data/dataB/datasets/MPI-Sintel_subset400/", photometric_augmentations=False),
    "kitti": KittiComb2015Train({},
                             "/data/dataB/datasets/KITTI_data_scene_flow",
                             photometric_augmentations=False,
                             preprocessing_crop=True)
}

base_out_path = "/data/dataA/model_interfaces/"

for model_name, model_path in models.items():
    print(f"recording interface for {model_name}")
    for dataset_name, dataset in datasets.items():
        print(f"dataset {dataset_name}")
        sample_interface_path = f"{base_out_path}{model_name}_{dataset_name}/"
        Path(sample_interface_path).mkdir(parents=True, exist_ok=True)

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
            x1_s["x1_"+ct_str] = x1.data.cpu().numpy()
            flow_s["flow_"+ct_str] = flow.data.cpu().numpy()
            l_s["l_"+ct_str] = np.array(l)
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
            clear_sample()
            model(sample_to_torch_batch(dataset[ii]))
            save_sample()

print("done")
