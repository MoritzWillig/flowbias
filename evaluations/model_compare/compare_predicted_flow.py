import math

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw

from flowbias.config import Config
from flowbias.losses import _elementwise_epe
from flowbias.utils.flow import compute_color, make_color_wheel
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch
from flowbias.utils.visualization.AGrid import AGrid

field = np.linspace([-1]*450, [+1]*450, num=450)
flow_wheel = compute_color(field, field.T)
flow_wheel[np.linalg.norm(np.dstack([field, field.T]), ord=2, axis=2)>0.99] = 1.0


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
    gt_flow = compute_color(gt_flow[:,:,0], gt_flow[:,:,1])/255


    results = []
    with torch.no_grad():
        for model_name in models:
            if model_name is None:
                results.append({"flow":torch.zeros((1,1))})
                continue
            model, transformer = load_model_from_meta(model_name)
            model.cuda().eval()
            results.append(model(transformer(sample)))


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
        gt_flow.shape,
        text_params={"font": font},
        title_params={"font": font})

    err_grid.title(eval_name)
    err_grid.place(0, 0, im1, "frame A")
    err_grid.place(1, 0, im2, "frame B")
    err_grid.place(2, 0, gt_flow, "ground truth")



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

        x = i % num_cols
        y = 1+(i//num_cols)
        label = model_name
        grid.place(x, y, compute_color(flow[0, 0, :,:], flow[0, 1, :,:])/255, label=label)
        err_grid.place(x, y, error_map, label=label)

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
comp_name = "chairs_on_middlebury"
models = ["pwc_chairs", "pwcWOX1_chairs", "pwc_on_CTSK", "pwcWOX1_on_CTSK"]
num_cols = 4

"""
evals = [
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
