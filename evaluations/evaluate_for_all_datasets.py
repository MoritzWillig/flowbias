import json
import sys

from flowbias.datasets.flyingchairs import FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanValid, FlyingThings3dCleanTrain
from flowbias.datasets.kitti_combined import KittiComb2015Val
from flowbias.datasets.sintel import SintelTrainingCleanValid, SintelTrainingFinalValid, SintelTrainingCleanFull, SintelTrainingFinalFull

from flowbias.models import PWCNet, FlowNet1S, PWCNetConv33Fusion, PWCNetX1Zero
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.losses import MultiScaleEPE_PWC, MultiScaleEPE_FlowNet
from flowbias.utils.statistics import SeriesStatistic

"""
Computes the average epe of a model for all datasets.

evaluate_for_all_datasets /path_to/model_checkpoint.ckpt networkName
"""

class ValArgs:
    def __init__(self):
        self.batch_size = 1


chairs_root = "/data/dataB/datasets/FlyingChairs_release/data/"
things_root = "/data/dataB/datasets/FlyingThings3D_subset/"
kitti_root = "/data/dataB/datasets/KITTI_data_scene_flow/"
sintel_clean_root = "/data/dataB/datasets/MPI-Sintel-complete/"
sintel_final_root = "/data/dataB/datasets/MPI-Sintel-complete/"

model_classes = {
    "pwc": [PWCNet, MultiScaleEPE_PWC],
    "flownet": [FlowNet1S, MultiScaleEPE_FlowNet],
    "pwcCon33": [PWCNetConv33Fusion, MultiScaleEPE_PWC],
    "pwcX1Zero": [PWCNetX1Zero, MultiScaleEPE_PWC]
}



#model_path = "/data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_latest.ckpt"
model_path = sys.argv[1]

if len(sys.argv) < 3:
    model_class_name = "pwc"
else:
    model_class_name = sys.argv[2]

model_class = model_classes[model_class_name][0]
loss_class = model_classes[model_class_name][1]

model = model_class(ValArgs())
load_model_parameters(model, model_path)
model.cuda()
loss = loss_class(ValArgs()).cuda()

datasets = {
    "flyingChairsValid": FlyingChairsValid({}, chairs_root, photometric_augmentations=False),
    "flyingChairsFull": FlyingChairsFull({}, chairs_root, photometric_augmentations=False),
    "flyingThingsCleanTrain": FlyingThings3dCleanTrain({}, things_root, photometric_augmentations=False),
    "flyingThingsCleanValid": FlyingThings3dCleanValid({}, things_root, photometric_augmentations=False),
    "kittiValid": KittiComb2015Val({}, kitti_root, photometric_augmentations=False),
    "sintelCleanValid": SintelTrainingCleanValid({}, sintel_clean_root, photometric_augmentations=False),
    "sintelCleanFull": SintelTrainingCleanFull({}, sintel_clean_root, photometric_augmentations=False),
    "sintelFinalValid": SintelTrainingFinalValid({}, sintel_final_root, photometric_augmentations=False),
    "sintelFinalFull": SintelTrainingFinalFull({}, sintel_final_root, photometric_augmentations=False)
}

model.eval()
loss.eval()

demo_sample = sample_to_torch_batch(datasets["flyingChairs"][0])
demo_loss_values = loss(model(demo_sample), demo_sample)
loss_names = list(demo_loss_values.keys())

results = {"model_path": model_path, "model_class_name": model_class_name}
for name, dataset in datasets.items():
    #print(f"dataset: {name}")

    losses = {name: SeriesStatistic() for name in loss_names}
    for i in range(len(dataset)):
        sample = sample_to_torch_batch(dataset[i])
        loss_values = loss(model(sample), sample)
        for lname, value in loss_values.items():
            losses[lname].push_value(value.cpu().detach().numpy())

    results[name] = {}
    for lname, lloss in losses.items():
        results[name][lname] = lloss.get_statistics(report_individual_values=False)


print(json.dumps(results))
#print("done")
