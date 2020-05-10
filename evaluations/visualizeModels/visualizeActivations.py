import math
import os

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm
import imageio

from flowbias.config import Config
from flowbias.losses import _elementwise_epe
from flowbias.utils.flow import compute_color, make_color_wheel
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch
from flowbias.utils.visualization.AGrid import AGrid

mappable = matplotlib.cm.ScalarMappable(cmap="PuOr")


def color_map(array):
    return mappable.to_rgba(array)[:,:,:3]

def deep_detach(output, _visited=None):
    """
    # break loops ...
    if _visited is None:
        _visited = set()

    oid = id(output)
    if oid in _visited:
        return
    else:
        _visited.add(oid)
    """

    if isinstance(output, torch.Tensor):
        return output.detach().cpu() #.numpy()
    elif isinstance(output, list):
        return [deep_detach(o, _visited) for o in output]
    if isinstance(output, dict):
        return {key: deep_detach(value, _visited) for key, value in output.items()}
    if isinstance(output, tuple):
        return [deep_detach(o, _visited) for o in output]
    else:
        print("unknown output type", type(output))
        return output

def unravel(output, _key="", tensor4=False):
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu()
        if tensor4:
            # tensor is a filter weight [out_channels, in_channels, *kernel_size]
            activation_container = []

            ocs, ics, kw, kh = output.size()
            kwi = kw + 1
            for oc in range(ocs):
                combined_filters = torch.ones((ics * kwi, kh, 3)).detach()
                for ic in range(ics):
                    combined_filters[ic*kwi:(ic+1)*kwi-1,:,:] = torch.from_numpy(color_map(output[oc, ic, :, :])).detach()
                activation_container.append((f"{_key}$out{oc}", combined_filters))
        else:
            # tensor is a activation map
            output = output.squeeze()
            activation_container = [(f"{_key}${idx}", output[idx]) for idx in range(output.size(0))]
    elif isinstance(output, list):
        activation_container = []
        for i, a in enumerate(output):
            activation_container.extend(unravel(a, f"{_key}${i}"))
    elif isinstance(output, dict):
        activation_container = []
        for key, a in output.items():
            activation_container.extend(unravel(a, f"{_key}${key}"))
    else:
        print("unexpected activation container", type(output), output)
        activation_container = []
    return activation_container

def do_evaluation(model, dataset_name, sample_id, dir_name):
    datasets = get_available_datasets(restrict_to=[dataset_name], force_mode="test")
    dataset = datasets[dataset_name]
    sample = sample_to_torch_batch(dataset[sample_id])

    im1 = sample["input1"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    im2 = sample["input2"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    gt_flow = sample["target1"]
    gt_flow = gt_flow.cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    gt_flow = compute_color(gt_flow[:,:,0], gt_flow[:,:,1])/255

    #with torch.no_grad():

    # assign the unique model name for all modules
    for name, module in model.named_modules():
        module._my_name = name
        module._my_iteration = 0

    # Visualize feature maps
    activations = {}
    trav_order = []
    def hook(module, input, output):
        print(module._my_name)
        # ignore double feature encoder activations
        if module._my_name not in activations:
            rt_name = module._my_name + "#" + str(module._my_iteration)
            module._my_iteration += 1

            trav_order.append(rt_name)
            activations[rt_name] = deep_detach(output)
        else:
            print("double activation", model._my_name)
    for name, module in model.named_modules():
        module.register_forward_hook(hook)

    #data, _ = dataset[0]
    #data.unsqueeze_(0)
    result = model(transformer(sample))

    for tr_id, tr_name in enumerate(trav_order):
        act = activations[tr_name]

        activation_container = unravel(act)
        min_w = min([a[1].shape[0] for a in activation_container])
        max_w = max([a[1].shape[0] for a in activation_container])
        min_h = min([a[1].shape[1] for a in activation_container])
        max_h = max([a[1].shape[1] for a in activation_container])

        ct = len(activation_container)
        if save_activations_img:
            grid_w = int(math.ceil(math.sqrt(ct)))
            grid_h = int(math.ceil(ct / grid_w))
            print(">>",activation_container[0][1].shape)
            grid = AGrid((grid_w, grid_h), (max_w, max_h), text_height=0, title_height=20)
            grid.title(tr_name)

            mn = +1000000
            mx = -1000000
            for (_, im) in activation_container:
                mn = min(torch.min(im), mn)
                mx = max(torch.max(im), mx)

            print(mn, mx)

            for idx, (name, im) in enumerate(activation_container):
                #mn = torch.min(im)
                #mx = torch.max(im)
                im_x = (im - mn) / (mx-mn)
                grid.place(idx % grid_w, idx // grid_w, im_x)

            imageio.imwrite(dir_name+str(tr_id)+"_"+tr_name+".png", grid.get_image())


        if save_activations_np:
            if (min_w != max_w) or (min_h != max_h):
                continue

            stacked_activations = np.zeros((ct, max_w, max_h))
            for idx, (name, im) in enumerate(activation_container):
                stacked_activations[idx, :, :] = im
            np.save(dir_name + "numpy/" + str(tr_id) + ".npy", stacked_activations)


def do_filters(model, dir_name):
    for param_name, param in model.named_parameters():
        if param_name.endswith("bias"):
            continue
        filter_container = unravel(param.detach().cpu(), _key=param_name, tensor4=True)

        max_w = max([a[1].shape[0] for a in filter_container])
        max_h = max([a[1].shape[1] for a in filter_container])

        ct = len(filter_container)
        grid_w = int(math.ceil(math.sqrt(ct)))
        grid_h = int(math.ceil(ct / grid_w))
        print(">>", param_name, filter_container[0][1].shape, param.shape)
        grid = AGrid((grid_w, grid_h), (max_w, max_h), text_height=0, title_height=20)
        grid.title(param_name)

        """
        mn = +1000000
        mx = -1000000
        for (_, im) in filter_container:
            mn = min(torch.min(im), mn)
            mx = max(torch.max(im), mx)
        """

        #print(mn, mx)

        for idx, (name, im) in enumerate(filter_container):
            mn = torch.min(im)
            mx = torch.max(im)
            im = (im - mn) / (mx - mn)
            #print(mn, mx)
            grid.place(idx % grid_w, idx // grid_w, im)

        imageio.imwrite(dir_name + param_name + ".png", grid.get_image())


activations_dir = "/data/dataA/temp/activations/"
filters_dir = "/data/dataA/temp/filters/"
#model_name = ["pwc_chairs", "pwcWOX1_chairs", "pwc_on_CTSK", "pwcWOX1_on_CTSK"]
#model_name = "pwc_on_CTSK"
#model_name = "pwcWOX1_on_CTSK"
model_name = "expertWOX1_CTSK_linAdd01_expert3"

save_activations_img = True
save_activations_np = True
save_activations = save_activations_img or save_activations_np
save_filters = False

evals = [
    {
        "dataset_name": "middleburyTrain",
        "sample_id": i
    } for i in [4]
]

model, transformer = load_model_from_meta(model_name)
model.cuda().eval()

if save_activations:
    for eval_data in evals:
        activations_dir_name = activations_dir + model_name + "_" + eval_data["dataset_name"] + "_" + str(eval_data["sample_id"]) + "/"

        os.makedirs(activations_dir_name, exist_ok=True)
        if save_activations_np:
            os.makedirs(activations_dir_name + "numpy/", exist_ok=True)
        do_evaluation(model, eval_data["dataset_name"], eval_data["sample_id"], activations_dir_name)

if save_filters:
    filters_dir_name = filters_dir + model_name + "/"

    os.makedirs(filters_dir_name, exist_ok=True)
    do_filters(model, filters_dir_name)