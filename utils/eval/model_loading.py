from collections import OrderedDict
import torch

from configuration import ModelAndLoss, CheckpointSaver


def transform_state_dict(state_dict, transform):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_state_dict[transform(key)] = value
    return new_state_dict


def prepare_sample(sample):
    for key, value in sample.items():
        if key.startswith("input") or key.startswith("target"):
            sample[key] = value.unsqueeze(0).cuda()
    return sample


def load_model_parameters(model_instance, checkpoint_path, strict=True):
    stats = torch.load(checkpoint_path)
    # remove "_model." from name
    model_params = transform_state_dict(stats["state_dict"], lambda name: name[7:])
    model_instance.load_state_dict(model_params, strict=strict)


def save_model(model, directory):
    resulting_model_and_loss = ModelAndLoss({}, model, None, None)
    checkpoint_saver = CheckpointSaver()
    stats_dict = dict(epe=float("inf"), epoch=0)
    checkpoint_saver.save_latest(directory, resulting_model_and_loss, stats_dict, True)


def sample_to_torch_batch(sample):
    for key, value in sample.items():
        if "input" in key or "target" in key:
            sample[key] = value.unsqueeze(0).cuda()
    return sample
