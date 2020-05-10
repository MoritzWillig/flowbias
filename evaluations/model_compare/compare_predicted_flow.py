import math

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
import matplotlib

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

flow_wheel = make_flow_wheel(450, 5)


def do_evaluation(comp_name, dataset_name, sample_id):
    data_name = f"{dataset_name}_frame{sample_id}"
    eval_name = f"{comp_name}_{data_name}"

    datasets = get_available_datasets(restrict_to=[dataset_name], force_mode="test")
    dataset = datasets[dataset_name]
    sample = sample_to_torch_batch(dataset[sample_id])

    im1 = sample["input1"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    im2 = sample["input2"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    gt_flow = sample["target1"]
    gt_flow = gt_flow.cpu().numpy()[0,:,:,:].transpose([1, 2, 0])
    #gt_flow = compute_color(gt_flow[:,:,0], gt_flow[:,:,1])/255

    #print("!!", data_name)
    gt_flow, max_sat = compute_color(gt_flow[:,:,0], gt_flow[:,:,1], None, True)


    results = []
    with torch.no_grad():
        for model_name in models:
            if model_name is None:
                results.append({"flow":torch.zeros((1,1))})
                continue
            model, transformer = load_model_from_meta(model_name)
            model.cuda().eval()
            results.append(model(transformer(sample)))

    if save_combined:
        #font = ImageFont.truetype("/home/moritz/.local/share/fonts/RobotoMono-Regular.ttf", 40, encoding="unic")
        font = ImageFont.truetype("/home/moritz/.local/share/fonts/RobotoMono-Bold.ttf", 40, encoding="unic")

        plt_size = (num_cols, 1+int(math.ceil(len(models)/num_cols)))
        max_shape = [
            max(flow_wheel.shape[0], gt_flow.shape[0]),
            max(flow_wheel.shape[1], gt_flow.shape[1])
        ]
        grid = AGrid(
            plt_size,
            max_shape,
            text_params={"font": font},
            title_params={"font": font})

        grid.title(eval_name)
        grid.place(0, 0, im1, "frame A")
        grid.place(1, 0, im2, "frame B")
        grid.place(2, 0, gt_flow, "ground truth")
        grid.place(3, 0, flow_wheel, "flow wheel")

        err_grid = AGrid(
            plt_size,
            [gt_flow.shape[0], gt_flow.shape[1]],
            text_params={"font": font},
            title_params={"font": font})

        err_grid.title(eval_name)
        err_grid.place(0, 0, im1, "frame A")
        err_grid.place(1, 0, im2, "frame B")
        err_grid.place(2, 0, gt_flow, "ground truth")
    else:
        save_im(im1, f"{Config.temp_directory}flow_compare/{comp_name}/{data_name}_imA.png")
        save_im(im2, f"{Config.temp_directory}flow_compare/{comp_name}/{data_name}_imB.png")
        save_im(gt_flow, f"{Config.temp_directory}flow_compare/{comp_name}/{data_name}_gt.png")
        save_im(flow_wheel, f"{Config.temp_directory}flow_compare/{comp_name}/{data_name}_wheel.png")

    gt = sample["target1"]
    max_errors = []
    for i, (model_name, result) in enumerate(zip(models, results)):
        fl = result["flow"]
        error_map = _elementwise_epe(fl, gt)
        im_max = float(torch.max(error_map).cpu().detach())
        max_errors.append(im_max)

    upper_50_percentile = sorted(max_errors)[math.ceil(min(len(max_errors)*0.50, len(max_errors)-1))]

    gt = sample["target1"]
    for i, (model_name, result) in enumerate(zip(models, results)):
        fl = result["flow"]

        error_map = _elementwise_epe(fl, gt)
        error_map = torch.clamp_max(error_map/upper_50_percentile, 1.0)
        error_map = error_map.squeeze().cpu().detach().numpy()

        flow = fl.cpu().detach().numpy()
        flow_rgb = compute_color(flow[0, 0, :,:], flow[0, 1, :,:], norm_sat=max_sat)

        if save_combined:
            x = i % num_cols
            y = 1+(i//num_cols)
            #grid.place(x, y, compute_color(flow[0, 0, :,:], flow[0, 1, :,:])/255, label=label)
            grid.place(x, y, flow_rgb, label=model_name)
            err_grid.place(x, y, error_map, label=model_name)
        else:
            name = data_name+"_"+model_name.replace(" ", "_")
            save_im(flow_rgb, f"{Config.temp_directory}flow_compare/{comp_name}/"+name+"_flow.png")
            save_im(error_map, f"{Config.temp_directory}flow_compare/{comp_name}/"+name+"_err.png")


    if save_combined:
        im = Image.fromarray(np.uint8(grid.get_image()*255), "RGB")
        im.save(f"{Config.temp_directory}flow_compare/{comp_name}/"+data_name+".png")

        im = Image.fromarray(np.uint8(err_grid.get_image()*255), "RGB")
        im.save(f"{Config.temp_directory}flow_compare/{comp_name}/"+data_name+"_err.png")



#models = ["A", "I", "H", "W"] # base models
#models = ["A", "F", "pwcWOX1_chairs", "expert_split02_expert0"] # chairs_base, chairs_fine, WOX1, expert_chairs
#models = [
#    "expert_split02_expert0", "expert_split02_expert1", "expert_split02_expert2", "expert_split02_expert3",
#    "expert_add01_expert0", "expert_add01_expert1", "expert_add01_expert2", "expert_add01_expert3",
#    "A", "pwc_chairs_iter_148", "I", "F",

#    "expert_add01_no_expert", "pwc_on_CTSK"
#]
#models = ["W", "pwcWOX1_kitti", "A", "S", "WOX1Blind_ks", "WOX1Blind_sk"]  # kitti networks

"""comp_name = "fused_models"
models = [
    "pwcWOX1_chairs", "WOX1Blind_ct", "WOX1Blind_cs", "WOX1Blind_ck",
    "WOX1Blind_tc", "pwcWOX1_things", "WOX1Blind_ts", "WOX1Blind_tk",
    "WOX1Blind_sc","WOX1Blind_st", "pwcWOX1_sintel", "WOX1Blind_sk",
    "WOX1Blind_kc", "WOX1Blind_kt", "WOX1Blind_ks", "pwcWOX1_kitti"
]
num_cols = 4"""
"""comp_name = "wox1_base_models"
models = [
    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti_temp"
]"""
#comp_name = "chairs_on_middlebury"
#models = ["pwc_chairs", "pwcWOX1_chairs", "pwc_on_CTSK", "pwcWOX1_on_CTSK"]
#models = ["pwc_chairs"]
#models = ["unifiedCTS_avg_expertWOX1_CTSK_add01_expert",
#          "expertWOX1_CTSK_add01_expert0", "expertWOX1_CTSK_add01_expert1", "expertWOX1_CTSK_add01_expert2", "expertWOX1_CTSK_add01_expert3"]
#comp_name = "pwc_and_wox1_middlebury"
#models = [
#    "pwc_chairs", "I", "H", "pwc_kitti",
#    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"
#]
comp_name = "secondaryFlow"
models = [ "pwcWOX1_secondary_flow_CTSK" ]
num_cols = 4
save_combined = False


"""evals = [
    {
        "dataset_name": "flyingChairsValid",
        "sample_id": 300
    },
    {
        "dataset_name": "flyingThingsCleanValid",
        "sample_id": 55
    },
    {
        "dataset_name": "sintelFinalValid",
        "sample_id": 130
    },
    {
        "dataset_name": "kitti2015Valid",
        "sample_id": 29
    }
]
"""
evals = [
    {
        "dataset_name": "middleburyTrain",
        "sample_id": i
    } for i in range(8)
]

for eval_data in evals:
    do_evaluation(comp_name, eval_data["dataset_name"], eval_data["sample_id"])
