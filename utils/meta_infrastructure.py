from collections import namedtuple
import argparse
import os

from flowbias.model_meta import model_meta, model_meta_fields, model_folders
import flowbias.models
from flowbias.utils.model_loading import load_model_parameters
from flowbias.config import Config

from flowbias.datasets.flyingchairs import FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanValid, FlyingThings3dCleanTrain
from flowbias.datasets.kitti_combined import KittiComb2015Train, KittiComb2015Val
from flowbias.datasets.sintel import SintelTrainingCleanValid, SintelTrainingFinalValid, SintelTrainingCleanFull, SintelTrainingFinalFull


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

    if load_latest:
        model_file = "checkpoint_latest.ckpt"
    else:
        model_file = "checkpoint_best.ckpt"
    return model_dir + "/" + model_file


class DataTransformer:
    def __init__(self, data):
        self._data = data

    def transform(self, sample):
        return {**sample, **self._data}

    def __call__(self, sample):
        return self.transform(sample)


def create_enricher(expert_id):
    return DataTransformer({"dataset": expert_id})


def create_no_enricher():
    return DataTransformer({})


loaders = {
    "_default": create_no_enricher(),
    "expert0": create_enricher(0),
    "expert1": create_enricher(1),
    "expert2": create_enricher(2),
    "expert3": create_enricher(3),
}


def load_model_from_meta(name, args=None, load_latest=False):
    meta = get_model_meta(name)
    model_path = assemble_meta_path(meta.folder_name, load_latest=load_latest)

    transformer = None
    if meta.loader_ is not None:
        transformer = loaders[meta.loader_]
    else:
        transformer = loaders["_default"]

    # create model instance
    if args is None:
        args = argparse.Namespace()
    module_dict = dict([(name, getattr(flowbias.models, name)) for name in dir(flowbias.models)])
    model = module_dict[meta.model](args)

    # load model parameters
    load_model_parameters(model, model_path)
    return model, transformer


def get_dataset_names():
    return [
        "flyingChairsValid", "flyingChairsFull",
        "flyingThingsCleanTrain", "flyingThingsCleanValid",
        "sintelCleanValid", "sintelCleanFull", "sintelFinalValid", "sintelFinalFull",
        "kitti2015Train", "kitti2015Valid"]


def get_available_datasets(force_mode=None, restrict_to=None):
    if restrict_to is None:
        restrict_to = get_dataset_names()

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
    if os.path.isdir(Config.dataset_locations["flyingChairs"]):
        if "flyingChairsValid" in restrict_to:
            available_datasets["flyingChairsValid"] = FlyingChairsValid({}, Config.dataset_locations["flyingChairs"], **params)
        if "flyingChairsFull" in restrict_to:
            available_datasets["flyingChairsFull"] = FlyingChairsFull({}, Config.dataset_locations["flyingChairs"], **params)
    if os.path.isdir(Config.dataset_locations["flyingThings"]):
        if "flyingThingsCleanTrain" in restrict_to:
            available_datasets["flyingThingsCleanTrain"] = FlyingThings3dCleanTrain({}, Config.dataset_locations["flyingThings"], **params)
        if "flyingThingsCleanValid" in restrict_to:
            available_datasets["flyingThingsCleanValid"] = FlyingThings3dCleanValid({}, Config.dataset_locations["flyingThings"], **params)
    if os.path.isdir(Config.dataset_locations["sintel"]):
        if "sintelCleanValid" in restrict_to:
            available_datasets["sintelCleanValid"] = SintelTrainingCleanValid({}, Config.dataset_locations["sintel"], **params)
        if "sintelCleanFull" in restrict_to:
            available_datasets["sintelCleanFull"] = SintelTrainingCleanFull({}, Config.dataset_locations["sintel"], **params)
        if "sintelFinalValid" in restrict_to:
            available_datasets["sintelFinalValid"] = SintelTrainingFinalValid({}, Config.dataset_locations["sintel"], **params)
        if "sintelFinalFull" in restrict_to:
            available_datasets["sintelFinalFull"] = SintelTrainingFinalFull({}, Config.dataset_locations["sintel"], **params)
    if os.path.isdir(Config.dataset_locations["kitti"]):
        if "kitti2015Train" in restrict_to:
            available_datasets["kitti2015Train"] = KittiComb2015Train({}, Config.dataset_locations["kitti"], **kitti_params)
        if "kitti2015Valid" in restrict_to:
            available_datasets["kitti2015Valid"] = KittiComb2015Val({}, Config.dataset_locations["kitti"], **kitti_params)
    return available_datasets


def dataset_needs_batch_size_one(dataset_name, force_mode=None):
    varying_image_sizes = []
    if force_mode is not "train":
        varying_image_sizes.append("kitti2015Valid")
    if force_mode == "test":
        varying_image_sizes.append("kitti2015Train")

    return dataset_name in varying_image_sizes
