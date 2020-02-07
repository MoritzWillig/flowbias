from collections import namedtuple
import argparse

from flowbias.model_meta import model_meta, model_meta_fields, model_folders
import flowbias.models
from flowbias.utils.model_loading import load_model_parameters

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


def load_model_from_meta(name, args=None, load_latest=False):
    meta = get_model_meta(name)
    model_path = assemble_meta_path(meta.folder_name, load_latest=load_latest)

    # create model instance
    if args is None:
        args = argparse.Namespace()
    module_dict = dict([(name, getattr(flowbias.models, name)) for name in dir(flowbias.models)])
    model = module_dict[meta.model](args)

    # load model parameters
    load_model_parameters(model, model_path)
    return model
