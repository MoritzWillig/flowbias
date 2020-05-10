import matplotlib.pyplot as plt
import time
from skimage.color import hsv2rgb

from flowbias.data_manipulation.extract_minor_flow import compute_secondary_flows
#from flowbias.utils.flow import compute_color
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch, stack_torch_batch
from flowbias.utils.visualization.AGrid import AGrid
import numpy as np

from PIL import Image

from flowbias.config import Config
from flowbias.losses import _elementwise_epe
from flowbias.utils.flow import compute_color, make_color_wheel
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch
from flowbias.utils.visualization.AGrid import AGrid

from skimage.color import hsv2rgb

def compute_color(x,y, norm_sat=None, report_norm_sat=False):
    mask = np.isnan(x)
    x[np.abs(x) > 1000] = np.NaN
    y[np.abs(y) > 1000] = np.NaN
    x = np.nan_to_num(x, nan=0)
    y = np.nan_to_num(y, nan=0)

    hue = np.arctan2(x, y)
    hue = ((hue + 2 * np.pi) % (2 * np.pi)) / (2*np.pi)

    sat = np.linalg.norm(np.dstack([x,y]), axis=2)
    if norm_sat is None:
        outlier_flow = np.ones_like(sat)
        #max_sat = np.max(sat) + 1e-8
        sat_sort = np.sort(sat.flat)
        max_sat = sat_sort[int(sat_sort.size*0.99)] + 1e-8
    else:
        outlier_flow = 1.0 - (sat > norm_sat) * 0.25
        max_sat = norm_sat
    sat /= max_sat
    sat = np.minimum(sat, 1.0)

    #print(np.min(hue), np.max(hue), np.min(sat), np.max(sat), np.min(outlier_flow), np.max(outlier_flow))

    #hsv = np.dstack([hue, sat, ~mask])
    hsv = np.dstack([hue, sat, outlier_flow])
    rgb = hsv2rgb(hsv)
    #rgb = np.dstack([sat, sat, sat])
    if report_norm_sat:
        return rgb, max_sat
    else:
        return rgb

def make_flow_wheel(size, ticks):
    field = np.linspace([-1]*450, [+1]*450, num=450)
    flow_wheel = compute_color(field, field.T, 1)
    #flow_wheel[np.linalg.norm(np.dstack([field, field.T]), ord=2, axis=2)>0.99] = 1.0
    return flow_wheel

def save_im(im, path):
    int_im = (im * 255).astype(np.uint8)
    Image.fromarray(int_im).save(path)


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
#dataset = get_available_datasets(restrict_to=["sintelCleanTrain"])["sintelCleanTrain"]
dataset = get_available_datasets(restrict_to=["middleburyTrain"])["middleburyTrain"]

for j in range(0, 10):
#for j in range(152, 500):
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

    xgt = gt[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0)
    gt_color = compute_color(xgt[:,:, 0], xgt[:,:, 1])
    save_im(gt_color, f"/data/dataA/temp/sec_flow/gt.png")

    print(f"gt shape: {gt.shape}")
    grid = AGrid([2, 6], [gt.shape[2], gt.shape[3]])
    grid.title(f"{j}")
    #grid.place(0, 0, compute_color(gt.squeeze().cpu().numpy().transpose(1,2,0)))
    #gt_rgb, gt_max = flow_to_rgb(gt[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), get_max=True)
    #grid.place(0, 0, gt_rgb)
    for i, (ma, mi) in enumerate(zip(major_levels[::-1], minor_levels[::-1])):
        if ma is None:
            print(f"{i} is none")
            continue
        print(f"{i} shape: {ma.shape}")

        #grid.place(0, i, flow_to_rgb(ma[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), max=gt_max))
        #grid.place(1, i, flow_to_rgb(mi[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0), max=gt_max))
        xma = ma[0, :, :, :].squeeze().cpu().numpy().transpose(1, 2, 0)
        xmi = mi[0, :, :, :].squeeze().cpu().numpy().transpose(1,2,0)
        ma_color = compute_color(xma[:,:, 0], xma[:,:, 1])
        mi_color = compute_color(xmi[:,:, 0], xmi[:,:, 1])
        grid.place(0, i, ma_color)
        grid.place(1, i, mi_color)

        save_im(ma_color, f"/data/dataA/temp/sec_flow/{j}_{i}_ma_flow.png")
        save_im(mi_color, f"/data/dataA/temp/sec_flow/{j}_{i}_mi_flow.png")

    #plt.imshow(grid.get_image())
    #plt.show()



