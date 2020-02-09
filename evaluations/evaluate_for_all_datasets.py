import json
import sys
import os
from datetime import datetime
import torch
import torch.utils.data as data

from flowbias.datasets.flyingchairs import FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanValid, FlyingThings3dCleanTrain
from flowbias.datasets.kitti_combined import KittiComb2015Train, KittiComb2015Val
from flowbias.datasets.sintel import SintelTrainingCleanValid, SintelTrainingFinalValid, SintelTrainingCleanFull, SintelTrainingFinalFull

from flowbias.models import PWCNet, FlowNet1S, PWCNetConv33Fusion, PWCNetX1Zero, PWCNetWOX1Connection, CTSKPWCExpertNet02
from flowbias.utils.meta_infrastructure import get_available_datasets, dataset_needs_batch_size_one
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.losses import MultiScaleEPE_PWC, MultiScaleEPE_FlowNet, MultiScaleSparseEPE_PWC, MultiScaleSparseEPE_FlowNet
from flowbias.utils.statistics import SeriesStatistic
from torch.utils.data.dataloader import DataLoader

"""
Computes the average epe of a model for all datasets.

evaluate_for_all_datasets /path_to/model_checkpoint.ckpt networkName
"""


class DataEnricher(data.Dataset):
    def __init__(self, dataset, additional):
        self._dataset = dataset
        self._additional = additional

    def __getitem__(self, index):
        return {**self._dataset[index], **self._additional}

    def __len__(self):
        return len(self._dataset)


class CTSKDatasetDetector(DataEnricher):
    # this are the dataset indices used by the CTSKTrain CombinedDataset and CTSKTrainDatasetBatchSampler
    _known_datasets = [
        [FlyingChairsValid, 0],
        [FlyingChairsFull, 0],
        [FlyingThings3dCleanValid, 1],
        [FlyingThings3dCleanTrain, 1],
        [SintelTrainingCleanValid, 2],
        [SintelTrainingFinalValid, 2],
        [SintelTrainingCleanFull, 2],
        [SintelTrainingFinalFull, 2],
        [KittiComb2015Train, 3],
        [KittiComb2015Val, 3],
    ]

    def _detect_dataset_id(self, dataset):
        dataset_id = -1
        for dataset_data in CTSKDatasetDetector._known_datasets:
            if isinstance(dataset, dataset_data[0]):
                #print("detected ", dataset_data[0])
                dataset_id = dataset_data[1]
        if dataset_id == -1:
            raise ValueError("Unknown dataset!")
        return dataset_id

    def __init__(self, dataset):
        super().__init__(dataset, {"dataset": self._detect_dataset_id(dataset)})


if __name__ == '__main__':
    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "preparing ...")

    class ValArgs:
        def __init__(self):
            self.batch_size = None

    model_classes = {
        "PWCNet": [PWCNet, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}],
        "FlowNet1S": [FlowNet1S, {"default": MultiScaleEPE_FlowNet, "kitti2015Train": MultiScaleSparseEPE_FlowNet, "kitti2015Valid": MultiScaleSparseEPE_FlowNet}],
        "PWCNetConv33Fusion": [PWCNetConv33Fusion, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}],
        "PWCNetX1Zero": [PWCNetX1Zero, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}],
        "PWCNetWOX1Connection": [PWCNetWOX1Connection, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}],
        "CTSKPWCExpertNet02Known": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNet02Expert0": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNet02Expert1": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNet02Expert2": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNet02Expert3": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]]
    }

    assert(len(sys.argv) == 4)

    #model_path = "/data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_latest.ckpt"
    model_path = sys.argv[1]
    model_class_name = sys.argv[2]
    result_file_path = sys.argv[3]
    print(model_path, "with", model_class_name)

    model_class = model_classes[model_class_name][0]
    model = model_class(ValArgs())
    load_model_parameters(model, model_path)
    model.eval().cuda()

    available_datasets = get_available_datasets(force_mode="test")

    rename = {
        "flyingChairs": "flyingChairsValid",
        "flyingThings": "flyingThingsCleanValid",
        "kitti": "kittiValid",
        "kittiValid": "kitti2015Valid",
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
                # check if the file contains old model_class_names
                if key == "model_class_name" and value not in model_class_name:
                    has_old_names = True
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
    #reevaluate = ["kitti2015Train", "kitti2015Valid"]  # forces datasets to be reevaluated
    reevaluate = []

    datasets = {
        dataset_name: dataset_data
        for dataset_name, dataset_data in available_datasets.items()
        if (dataset_name not in existing_results_datasets) or (dataset_name in reevaluate)}

    print("available_datasets:", list(available_datasets.keys()))
    print("existing results:", list(existing_results.keys()))
    print("computing results for:", list(datasets.keys()))

    if len(datasets.keys()) == 0:
        if has_old_names:
            print("replacing old dataset or model names")
            results = {"model_path": model_path, "model_class_name": model_class_name}
            for key, value in existing_results.items():
                results[key] = value
            with open(result_file_path, "w") as f:
                f.write(json.dumps(results))
        print("no datasets remaining - exiting")
        exit()

    batch_size = 16

    with torch.no_grad():
        model_config = model_classes[model_class_name]

        demo_available_dataset = next(iter(datasets.values()))
        if len(model_config) > 2:
            # wrap dataset into dataset enricher
            enricherConfig = model_config[2]
            demo_available_dataset = enricherConfig[0](demo_available_dataset, enricherConfig[1])
        demo_sample = sample_to_torch_batch(demo_available_dataset[0])
        demo_loss = model_classes[model_class_name][1]["default"](ValArgs()).eval().cuda()
        demo_loss_values = demo_loss(model(demo_sample), demo_sample)
        loss_names = list(demo_loss_values.keys())

        results = {"model_path": model_path, "model_class_name": model_class_name}

        for name, dataset in datasets.items():
            print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), name)

            loss_class = model_config[1][name] if name in model_config[1] else model_config[1]["default"]
            loss = loss_class(ValArgs()).eval().cuda()
            if len(model_config) > 2:
                # wrap dataset into dataset enricher
                enricherConfig = model_config[2]
                dataset = enricherConfig[0](dataset, enricherConfig[1])

            losses = {name: SeriesStatistic() for name in loss_names}
            dataset_size = len(dataset)
            i = 0

            gpuargs = {"num_workers": 4, "pin_memory": False}
            loader = DataLoader(
                dataset,
                batch_size=1 if dataset_needs_batch_size_one(name, force_mode="test") else batch_size,
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
        # but keep newer results (in case we reevaluated a dataset)
        if key in results:
            continue
        results[key] = value

    # save
    with open(result_file_path, "w") as f:
        f.write(json.dumps(results))

    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "done")
