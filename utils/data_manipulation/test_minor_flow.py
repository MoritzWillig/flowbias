import matplotlib.pyplot as plt
import time
from skimage.color import hsv2rgb

from flowbias.data_manipulation.extract_minor_flow import compute_secondary_flows
from flowbias.utils.flow import compute_color
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch, stack_torch_batch
from flowbias.utils.visualization.AGrid import AGrid
import numpy as np


def flow_to_rgb(flow, max=None, get_max=False):
    if get_max:
        return compute_color(flow), None
    else:
        return compute_color(flow)

    mask = np.isnan(flow)
    flow = np.nan_to_num(flow, nan=0)

    hue = np.arctan2(flow[:, :, 0], flow[:, :, 1])
    hue = ((hue + 2 * np.pi) % (2 * np.pi)) / (2*np.pi)

    sat = np.linalg.norm(flow, axis=2)
    if max is None:
        max = np.max(sat) + 1e-8
    sat /= max
    hsv = np.dstack([hue, sat, ~mask[:,:,0]])
    rgb = hsv2rgb(hsv)
    if get_max:
        return rgb, max
    else:
        return rgb


num_iterations = 5
#dataset = get_available_datasets(restrict_to=["flyingThingsCleanTrain"])["flyingThingsCleanTrain"]
#dataset = get_available_datasets(restrict_to=["flyingChairsTrain"])["flyingChairsTrain"]
dataset = get_available_datasets(restrict_to=["sintelCleanTrain"])["sintelCleanTrain"]
#dataset = get_available_datasets(restrict_to=["middleburyTrain"])["middleburyTrain"]

#for j in range(0, 10):
for j in range(152, 500):
    image = stack_torch_batch(sample_to_torch_batch(dataset[j]))

    #model, transformer = load_model_from_meta("pwcWOX1_on_CTSK")
    #model.cuda().train()
    #pred = model(transformer(image))
    #print([i.shape for i in pred["flow"]])


    gt = image["target1"]

    start = time.clock()
    num_test_iters = 100
    for i in range(num_test_iters):
        major_levels, minor_levels = compute_secondary_flows(gt, num_iterations)
    end = time.clock()
    print("time per computation:", 1000*(end - start)/num_test_iters, "ms")

    print(f"gt shape: {gt.shape}")
    grid = AGrid([2, 6], [gt.shape[2], gt.shape[3]])
    grid.title(f"{j}")
    #grid.place(0, 0, compute_color(gt.squeeze().cpu().numpy().transpose(1,2,0)))
    gt_rgb, gt_max = flow_to_rgb(gt[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), get_max=True)
    grid.place(0, 0, gt_rgb)
    for i, (ma, mi) in enumerate(zip(major_levels[::-1], minor_levels[::-1])):
        if ma is None:
            print(f"{i} is none")
            continue
        print(f"{i} shape: {ma.shape}")

        grid.place(0, i, flow_to_rgb(ma[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), max=gt_max))
        grid.place(1, i, flow_to_rgb(mi[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), max=gt_max))

    plt.imshow(grid.get_image())
    plt.show()

