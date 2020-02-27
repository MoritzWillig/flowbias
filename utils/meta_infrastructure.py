from collections import namedtuple
import argparse
import os
import csv

from flowbias.datasets.subsampledDataset import SubsampledDataset
from flowbias.model_meta import model_meta, model_meta_fields, model_folders
import flowbias.models
from flowbias.utils.model_loading import load_model_parameters
from flowbias.config import Config

from flowbias.datasets.flyingchairs import FlyingChairsTrain, FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanTrain, FlyingThings3dCleanValid, \
    FlyingThings3dCleanFull, FlyingThings3d
from flowbias.datasets.kitti_combined import KittiComb2015Train, KittiComb2015Val, KittiComb2015Full
from flowbias.datasets.sintel import SintelTrainingCleanTrain, SintelTrainingCleanValid, SintelTrainingCleanFull
from flowbias.datasets.sintel import SintelTrainingFinalTrain, SintelTrainingFinalValid, SintelTrainingFinalFull
from flowbias.losses import MultiScaleEPE_PWC, MultiScaleEPE_FlowNet, MultiScaleSparseEPE_PWC, MultiScaleSparseEPE_FlowNet
from flowbias.models import PWCNet, PWCNetWOX1Connection, PWCNetWOX1ConnectionExt, PWCNetConv33Fusion, PWCExpertNet, PWCExpertAddNet, CTSKPWCExpertNetAdd01, CTSKPWCExpertNet02, PWCNetDSEncoder
from flowbias.models import FlowNet1S

MetaEntry = namedtuple("MetaEntry", model_meta_fields)


def get_model_meta(name):
    if isinstance(name, MetaEntry):
        return name
    if not (name in model_meta):
        raise ValueError(f"unknown model name: {name}")
    return MetaEntry(*model_meta[name])


def assemble_meta_path(meta_path, load_latest):
    splits = meta_path.split("@", 1)
    if len(splits) == 1:
        base_name = "_default"
    else:
        base_name = splits[0]
    model_dir = model_folders[base_name] + splits[-1]

    if model_dir.endswith(".ckpt"):
        model_file = ""
    else:
        if load_latest:
            model_file = "/checkpoint_latest.ckpt"
        else:
            model_file = "/checkpoint_best.ckpt"
    return model_dir + model_file


class DataTransformer:
    def __init__(self, data):
        self._data = data

    def transform(self, sample):
        return {**sample, **self._data}

    def __call__(self, sample):
        return self.transform(sample)


class NoTransformer:
    def __call__(self, sample):
        return sample


def create_enricher(expert_id):
    return DataTransformer({"dataset": expert_id})


def create_no_enricher():
    return NoTransformer()


loaders = {
    "_default": create_no_enricher(),
    "noExpert": create_enricher(-1),
    "expert0": create_enricher(0),
    "expert1": create_enricher(1),
    "expert2": create_enricher(2),
    "expert3": create_enricher(3),
}


def load_model_from_meta(name, args=None, force_architecture=None, load_latest=False, strict_parameter_loading=None):
    meta = get_model_meta(name)
    model_path = assemble_meta_path(meta.folder_name, load_latest=load_latest)

    transformer = None
    if meta.loader_ is not None:
        transformer = loaders[meta.loader_]
    else:
        transformer = loaders["_default"]

    # create model instance
    if args is None:
        args = {"args": argparse.Namespace()}
    module_dict = dict([(name, getattr(flowbias.models, name)) for name in dir(flowbias.models)])

    if force_architecture is None:
        architecture = meta.model
    else:
        architecture = force_architecture
    model = module_dict[architecture](**args)

    # load model parameters
    loading_parameters = {"strict": strict_parameter_loading} if strict_parameter_loading is not None else {}
    load_model_parameters(model, model_path, **loading_parameters)
    return model, transformer


dataset_splits = [
    [ "flyingChairsTrain", ["flyingChairs", "train"]],
    [ "flyingChairsValid", ["flyingChairs", "valid"]],
    [ "flyingChairsFull", ["flyingChairs", "full"]],
    [ "flyingThingsCleanTrain", ["flyingThings", "train"]],
    [ "flyingThingsCleanValid", ["flyingThings", "valid"]],
    [ "flyingThingsCleanFull", ["flyingThings", "full"]],
    [ "sintelCleanTrain", ["sintel", "train", "clean"]],
    [ "sintelCleanValid", ["sintel", "valid", "clean"]],
    [ "sintelCleanFull", ["sintel", "full", "clean"]],
    [ "sintelFinalTrain", ["sintel", "train", "final"]],
    [ "sintelFinalValid", ["sintel", "valid", "final"]],
    [ "sintelFinalFull", ["sintel", "full", "final"]],
    [ "kitti2015Train", ["kitti", "train"]],
    [ "kitti2015Valid", ["kitti", "valid"]],
    [ "kitti2015Full", ["kitti", "full"]]
]

dataset_sets = {
    "subsets": [
        ["flyingChairsSubset", ["flyingChairs", "subset"]],
        ["flyingThingsSubset", ["flyingThings", "subset"]],
        ["sintelSubset", ["sintel", "subset"]],
        ["kittiSubset", ["kitti", "subset"]],  # since kitti has so little data, this is equal to kitti train
    ]
}


def create_any_selector(candidates):
    candidates = set(candidates)
    return lambda tags: not candidates.isdisjoint(tags)


def create_filter_selector(exceptions):
    exceptions = set(exceptions)
    return lambda tags: exceptions.isdisjoint(tags)


def get_dataset_names(select_by_any_tag=None, exclude_by_tag=None, datasets="main"):
    if datasets is "main":
        dataset_set = dataset_splits
    else:
        dataset_set = dataset_sets[datasets]

    selectors = []
    if select_by_any_tag is not None:
        selectors.append(create_any_selector(select_by_any_tag))

    if exclude_by_tag is not None:
        selectors.append(create_filter_selector(exclude_by_tag))
    return [dataset[0] for dataset in dataset_set if all([selector(dataset[1]) for selector in selectors])]


def get_available_datasets(force_mode=None, restrict_to=None, select_by_any_tag=None, exclude_by_tag=None, datasets="main"):
    if restrict_to is not None and select_by_any_tag is not None:
        ValueError("restrict_to and select_by_tag parameters are mutually exclusive")

    if restrict_to is None:
        restrict_to = get_dataset_names(select_by_any_tag=select_by_any_tag, exclude_by_tag=exclude_by_tag, datasets=datasets)

    if force_mode is None:
        params = {}
        kitti_params = {}
    elif force_mode == "train":
        params = {"photometric_augmentations": True}
        kitti_params = {"preprocessing_crop": True, **params}
    elif force_mode == "test":
        params = {"photometric_augmentations": False}
        kitti_params = {"preprocessing_crop": False, **params}
    else:
        raise ValueError("unknown dataset mode")

    available_datasets = {}
    if datasets == "main":
        _get_available_main_split(available_datasets, restrict_to, params, kitti_params)
    else:
        _get_available_sub_split(datasets, available_datasets, restrict_to, params, kitti_params)
    return available_datasets


def _get_available_main_split(available_datasets, restrict_to, params, kitti_params):
    if os.path.isdir(Config.dataset_locations["flyingChairs"]):
        if "flyingChairsTrain" in restrict_to:
            available_datasets["flyingChairsTrain"] = FlyingChairsTrain({}, Config.dataset_locations["flyingChairs"], **params)
        if "flyingChairsValid" in restrict_to:
            available_datasets["flyingChairsValid"] = FlyingChairsValid({}, Config.dataset_locations["flyingChairs"], **params)
        if "flyingChairsFull" in restrict_to:
            available_datasets["flyingChairsFull"] = FlyingChairsFull({}, Config.dataset_locations["flyingChairs"], **params)
    if os.path.isdir(Config.dataset_locations["flyingThings"]):
        if "flyingThingsCleanTrain" in restrict_to:
            available_datasets["flyingThingsCleanTrain"] = FlyingThings3dCleanTrain({}, Config.dataset_locations["flyingThings"], **params)
        if "flyingThingsCleanValid" in restrict_to:
            available_datasets["flyingThingsCleanValid"] = FlyingThings3dCleanValid({}, Config.dataset_locations["flyingThings"], **params)
        if "flyingThingsCleanFull" in restrict_to:
            available_datasets["flyingThingsCleanFull"] = FlyingThings3dCleanFull({}, Config.dataset_locations["flyingThings"], **params)
    if os.path.isdir(Config.dataset_locations["sintel"]):
        if "sintelCleanTrain" in restrict_to:
            available_datasets["sintelCleanTrain"] = SintelTrainingCleanTrain({}, Config.dataset_locations["sintel"], **params)
        if "sintelCleanValid" in restrict_to:
            available_datasets["sintelCleanValid"] = SintelTrainingCleanValid({}, Config.dataset_locations["sintel"], **params)
        if "sintelCleanFull" in restrict_to:
            available_datasets["sintelCleanFull"] = SintelTrainingCleanFull({}, Config.dataset_locations["sintel"], **params)
        if "sintelFinalTrain" in restrict_to:
            available_datasets["sintelFinalTrain"] = SintelTrainingFinalTrain({}, Config.dataset_locations["sintel"], **params)
        if "sintelFinalValid" in restrict_to:
            available_datasets["sintelFinalValid"] = SintelTrainingFinalValid({}, Config.dataset_locations["sintel"], **params)
        if "sintelFinalFull" in restrict_to:
            available_datasets["sintelFinalFull"] = SintelTrainingFinalFull({}, Config.dataset_locations["sintel"], **params)
    if os.path.isdir(Config.dataset_locations["kitti"]):
        if "kitti2015Train" in restrict_to:
            available_datasets["kitti2015Train"] = KittiComb2015Train({}, Config.dataset_locations["kitti"], **kitti_params)
        if "kitti2015Valid" in restrict_to:
            available_datasets["kitti2015Valid"] = KittiComb2015Val({}, Config.dataset_locations["kitti"], **kitti_params)
        if "kitti2015Full" in restrict_to:
            available_datasets["kitti2015Full"] = KittiComb2015Full({}, Config.dataset_locations["kitti"], **kitti_params)


def _get_available_sub_split(dataset_set, available_datasets, restrict_to, params, kitti_params):
    if dataset_set != "subsets":
        raise ValueError(f"Unknown dataset set '{dataset_set}'")

    if os.path.isdir(Config.dataset_locations["flyingChairsSubset"]) and ("flyingChairsSubset" in restrict_to):
        available_datasets["flyingChairsSubset"] = FlyingChairsFull({}, Config.dataset_locations["flyingChairsSubset"], **params)
    if os.path.isdir(Config.dataset_locations["flyingThingsSubset"]) and ("flyingThingsSubset" in restrict_to):
        available_datasets["flyingThingsSubset"] = FlyingThings3d(
            {},
            Config.dataset_locations["flyingThingsSubset"]+"/train/image_clean/left",
            Config.dataset_locations["flyingThingsSubset"]+"/train/flow/left", "", **params)
    if os.path.isdir(Config.dataset_locations["sintelSubset"]) and ("sintelSubset" in restrict_to):
        available_datasets["sintelSubset"] = SubsampledDataset({}, Config.dataset_locations["sintelSubset"], **params)
    if os.path.isdir(Config.dataset_locations["kittiSubset"]) and ("kittiSubset" in restrict_to):
        available_datasets["kittiSubset"] = KittiComb2015Train({}, Config.dataset_locations["kittiSubset"], **kitti_params)


def switch_to_train(loss):
    return loss.eval()


def switch_to_eval(loss):
    return loss.eval()


def get_loss(loss_name, model_instance, dataset_name, loss_args=None):
    losses = {
        "epe": [
            {
                "models": [
                        PWCNet, PWCNetWOX1Connection, PWCNetWOX1ConnectionExt, PWCNetConv33Fusion, PWCExpertNet,
                        PWCExpertAddNet, CTSKPWCExpertNetAdd01, CTSKPWCExpertNet02, PWCNetDSEncoder
                    ],
                "losses": {
                    "@dense": MultiScaleEPE_PWC,
                    "@sparse": MultiScaleSparseEPE_PWC
                },
                "processor": switch_to_train
            },
            {
                "models": [ FlowNet1S ],
                "losses": [ MultiScaleEPE_FlowNet, MultiScaleSparseEPE_FlowNet],
                "processor": switch_to_eval
            }
        ],
        "total_loss": [
            {
                "models": [
                    PWCNet, PWCNetWOX1Connection, PWCNetWOX1ConnectionExt, PWCNetConv33Fusion, PWCExpertNet,
                    PWCExpertAddNet, CTSKPWCExpertNetAdd01, CTSKPWCExpertNet02, PWCNetDSEncoder
                ],
                "losses": {
                    "@dense": MultiScaleEPE_PWC,
                    "@sparse": MultiScaleSparseEPE_PWC
                },
                "processor": switch_to_train
            },
            {
                "models": [FlowNet1S],
                "losses": [MultiScaleEPE_FlowNet, MultiScaleSparseEPE_FlowNet],
                "processor": switch_to_train
            }
        ]
    }

    # select a list of losses, that can handle the model output
    loss_candidates = losses[loss_name]
    selected_candidate = None
    for candidate in loss_candidates:
        if model_instance.__class__ in candidate["models"]:
            selected_candidate = candidate
            break

    if selected_candidate is None:
        raise ValueError("no loss found for handling the given model output")

    # find category of the dataset
    dataset_categories = {
        "_default": "@dense",
        "KITTI": "@sparse"
    }

    if dataset_name in dataset_categories:
        category = dataset_categories[dataset_name]
    else:
        category = dataset_categories["_default"]

    if category not in selected_candidate["losses"]:
        raise ValueError("no loss found for the given dataset category")

    loss = selected_candidate["losses"]
    processor = selected_candidate["processor"]

    if loss_args is None:
        loss_args = {}

    return processor(loss(**loss_args))


def dataset_needs_batch_size_one(dataset_name, force_mode=None):
    varying_image_sizes = []
    # kitti2015Valid does only crop in train mode:
    if force_mode != "train":
        varying_image_sizes.append("kitti2015Valid")

    # if we force test mode, we do not crop the images
    if force_mode == "test":
        varying_image_sizes.append("kitti2015Train")
        varying_image_sizes.append("kitti2015Full")

    return dataset_name in varying_image_sizes


def get_eval_summary():
    summary = {}

    with open(Config.eval_summary_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            summary[row["model_id"]] = dict(row)
    return summary
