import json
import sys
import os
from datetime import datetime
import torch
import time

from flowbias.datasets.flyingchairs import FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanValid, FlyingThings3dCleanTrain
from flowbias.datasets.kitti_combined import KittiComb2015Val
from flowbias.datasets.sintel import SintelTrainingCleanValid, SintelTrainingFinalValid, SintelTrainingCleanFull, SintelTrainingFinalFull

from flowbias.models import PWCNet, FlowNet1S, PWCNetConv33Fusion, PWCNetX1Zero
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.losses import MultiScaleEPE_PWC, MultiScaleEPE_FlowNet
from flowbias.utils.statistics import SeriesStatistic
from torch.utils.data.dataloader import DataLoader

from flowbias.config import Config

"""
Computes the average epe of a model for all datasets.

evaluate_for_all_datasets /path_to/model_checkpoint.ckpt networkName
"""

if __name__ == '__main__':
    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "preparing ...")

    class ValArgs:
        def __init__(self):
            self.batch_size = None

    model_classes = {
        "pwc": [PWCNet, MultiScaleEPE_PWC],
        "flownet": [FlowNet1S, MultiScaleEPE_FlowNet],
        "pwcConv33": [PWCNetConv33Fusion, MultiScaleEPE_PWC],
        "pwcX1Zero": [PWCNetX1Zero, MultiScaleEPE_PWC]
    }

    assert(len(sys.argv) == 4)

    #model_path = "/data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_latest.ckpt"
    model_path = sys.argv[1]
    model_class_name = sys.argv[2]
    result_file_path = sys.argv[3]
    print(model_path, "with", model_class_name)

    model_class = model_classes[model_class_name][0]
    loss_class = model_classes[model_class_name][1]

    model = model_class(ValArgs())
    load_model_parameters(model, model_path)
    model.cuda()
    loss = loss_class(ValArgs()).cuda()

    available_datasets = {}
    if os.path.isdir(Config.dataset_locations["flyingChairs"]):
        available_datasets["flyingChairsValid"] = FlyingChairsValid({}, Config.dataset_locations["flyingChairs"], photometric_augmentations=False)
        available_datasets["flyingChairsFull"] = FlyingChairsFull({}, Config.dataset_locations["flyingChairs"], photometric_augmentations=False)
    if os.path.isdir(Config.dataset_locations["flyingThings"]):
        available_datasets["flyingThingsCleanTrain"] = FlyingThings3dCleanTrain({}, Config.dataset_locations["flyingThings"], photometric_augmentations=False)
        available_datasets["flyingThingsCleanValid"] = FlyingThings3dCleanValid({}, Config.dataset_locations["flyingThings"], photometric_augmentations=False)
    if os.path.isdir(Config.dataset_locations["kitti"]):
        available_datasets["kittiValid"] = KittiComb2015Val({}, Config.dataset_locations["kitti"], photometric_augmentations=False)
    if os.path.isdir(Config.dataset_locations["sintel"]):
        available_datasets["sintelCleanValid"] = SintelTrainingCleanValid({}, Config.dataset_locations["sintel"], photometric_augmentations=False)
        available_datasets["sintelCleanFull"] = SintelTrainingCleanFull({}, Config.dataset_locations["sintel"], photometric_augmentations=False)
        available_datasets["sintelFinalValid"] = SintelTrainingFinalValid({}, Config.dataset_locations["sintel"], photometric_augmentations=False)
        available_datasets["sintelFinalFull"] = SintelTrainingFinalFull({}, Config.dataset_locations["sintel"], photometric_augmentations=False)

    need_batch_size_one = ["kittiValid"]

    rename = {
        "flyingChairs": "flyingChairsValid",
        "flyingThings": "flyingThingsCleanValid",
        "kitti": "kittiValid",
        "sintelClean": "sintelCleanValid",
        "sintelFinal": "sintelFinalValid",
    }

    # load existing results
    has_old_names = False
    if os.path.isfile(result_file_path):
        with open(result_file_path, "r") as f:
            existing_results_x = json.loads(f.read())

        # rename old keys and skip non-dataset entries
        existing_results = {}
        for key, value in existing_results_x.items():
            if key in ["model_path", "model_class_name"]:
                continue

            if key in rename:
                existing_results[rename[key]] = value
                has_old_names = True
            else:
                existing_results[key] = value
    else:
        # no existing results
        existing_results = {}
    existing_results_datasets = list(existing_results.keys())

    # compute remaining evaluations
    datasets = {dataset_name: dataset_data for dataset_name, dataset_data in available_datasets.items() if dataset_name not in existing_results_datasets}

    print("available_datasets:", list(available_datasets.keys()))
    print("existing results:", list(existing_results.keys()))
    print("computing results for:", list(datasets.keys()))

    if len(datasets.keys()) == 0:
        if has_old_names:
            print("replacing old dataset names")
            results = {"model_path": model_path, "model_class_name": model_class_name}
            for key, value in existing_results.items():
                results[key] = value
            with open(result_file_path, "w") as f:
                f.write(json.dumps(results))
        print("no datasets remaining - exiting")
        exit()


    model.eval()
    loss.eval()

    batch_size = 16

    with torch.no_grad():
        demo_available_dataset = next(iter(datasets.values()))
        demo_sample = sample_to_torch_batch(demo_available_dataset[0])
        demo_loss_values = loss(model(demo_sample), demo_sample)
        loss_names = list(demo_loss_values.keys())

        results = {"model_path": model_path, "model_class_name": model_class_name}

        for name, dataset in datasets.items():
            print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), name)

            losses = {name: SeriesStatistic() for name in loss_names}
            dataset_size = len(dataset)
            i = 0

            gpuargs = {"num_workers": 4, "pin_memory": False}
            #gpuargs = {"pin_memory": False}
            loader = DataLoader(
                dataset,
                batch_size=batch_size if name not in need_batch_size_one else 1,
                shuffle=False,
                drop_last=False,
                **gpuargs)

            #for i in range(len(dataset)):
            for sample in loader:
                input_keys = list(filter(lambda x: "input" in x, sample.keys()))
                target_keys = list(filter(lambda x: "target" in x, sample.keys()))
                tensor_keys = input_keys + target_keys

                for key, value in sample.items():
                    if key in tensor_keys:
                        sample[key] = value.cuda(non_blocking=True)

                loss_values = loss(model(sample), sample)
                for lname, value in loss_values.items():
                    b, _, _, _ = sample["target1"].size()
                    losses[lname].push_value(value.cpu().detach().numpy(), int(b))

                #time.sleep(0.003)

                #i += 1
                #if i+1 % 10 == 0:
                #    sys.stdout.write(f"\r{i}/{dataset_size}")
                #    sys.stdout.flush()
            #sys.stdout.write("\n")
            #sys.stdout.flush()

            results[name] = {}
            for lname, lloss in losses.items():
                statistic = lloss.get_statistics(report_individual_values=False)
                results[name][lname] = statistic
                print(f"{lname}: {statistic['average']}")

    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "saving ...")

    # add existing results
    for key, value in existing_results.items():
        results[key] = value

    # save
    with open(result_file_path, "w") as f:
        f.write(json.dumps(results))

    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "done")
