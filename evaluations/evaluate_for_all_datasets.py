import json
import sys

from flowbias.datasets.flyingchairs import FlyingChairsValid
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanValid
from flowbias.datasets.kitti_combined import KittiComb2015Val
from flowbias.datasets.sintel import SintelTrainingCleanValid

from flowbias.models import PWCNet
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.losses import MultiScaleEPE_PWC
from flowbias.utils.statistics import SeriesStatistic


class ValArgs:
    def __init__(self):
        self.batch_size = 1


chairs_root = "/data/dataB/datasets/FlyingChairs_release/data/"
things_root = "/data/dataB/datasets/FlyingThings3D_subset/"
kitti_root = "/data/dataB/datasets/KITTI_data_scene_flow/"
sintel_root = "/data/dataB/datasets/MPI-Sintel-complete/"

#model_path = "/data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_latest.ckpt"
model_path = sys.argv[1]


model = PWCNet(ValArgs)
load_model_parameters(model, model_path)
model.cuda()
loss = MultiScaleEPE_PWC(ValArgs()).cuda()

datasets = {
    "flyingChairs": FlyingChairsValid({}, chairs_root, photometric_augmentations=False),
    "flyingThings": FlyingThings3dCleanValid({}, things_root, photometric_augmentations=False),
    "kitti": KittiComb2015Val({}, kitti_root, photometric_augmentations=False),
    "sintel": SintelTrainingCleanValid({}, sintel_root, photometric_augmentations=False)
}

model.eval()
loss.eval()

demo_sample = sample_to_torch_batch(datasets["flyingChairs"][0])
demo_loss_values = loss(model(demo_sample), demo_sample)
loss_names = list(demo_loss_values.keys())

results = {"model_path": model_path}
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
        results[name][lname] = lloss.get_statistics()


print(json.dumps(results))
#print("done")
