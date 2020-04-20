import os
from datetime import datetime
import torch
import numpy as np
import math

from flowbias.losses import _elementwise_epe, _upsample2d_as
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch
from flowbias.utils.visualization.AGrid import AGrid

import matplotlib.pyplot as plt

produce_base_images = False
produce_border_distance_images = True
produce_center_distance_images = True


spatialAEPE_dir = "/data/dataB/temp/spatialAEPE/"
model_names = [
    "expertWOX1_CTSK_add01_expert0", "expertWOX1_CTSK_add01_expert1", "expertWOX1_CTSK_add01_expert2", "expertWOX1_CTSK_add01_expert3",
    "pwcWOX1_on_CTSK",
    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti",
    "pwc_chairs", "I", "H", "pwc_kitti"
    ]


def compute_average_edge_distance_value(a):
    w = a.shape[0]
    h = a.shape[1]
    side_min = math.ceil(min(w, h) / 2)
    #side_min = 200

    avgs = np.zeros(side_min)
    for d in range(side_min):
        dd = d + 1
        l = np.sum(a[d, d:-dd])
        r = np.sum(a[-dd, d:-dd])
        b = np.sum(a[d:-dd, d])
        t = np.sum(a[d:-dd, -dd])

        ebl = a[d,d]
        etl = a[d,-dd]
        ebr = a[-dd,d]
        etr = a[-dd,-dd]

        w_side = w - (2*d)
        h_side = h - (2*d)
        total = 2 * w_side + 2 * h_side - 4

        avg = (l + r + b + t - (ebl + etl + ebr + etr)) / total
        avgs[d] = avg
    return avgs


datasets = get_available_datasets(select_by_any_tag=["valid"], force_mode="test", run_dry=True)
with torch.no_grad():
    for model_name in model_names:
        model, transformer = load_model_from_meta(model_name)
        model.cuda().eval()

        print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), f"evaluating {model_name}")

        for dataset_name in datasets:
            print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), f"dataset {dataset_name}")
            model_spatialAEPE_dir = spatialAEPE_dir + dataset_name + "_" + model_name + "/"

            if not os.path.exists(model_spatialAEPE_dir + "spatial_aepe.npy"):
                os.makedirs(model_spatialAEPE_dir, exist_ok=True)

                dataset = get_available_datasets(restrict_to=[dataset_name], force_mode="test")
                print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), f"loading dataset: {len(dataset)} samples")
                demo_sample = sample_to_torch_batch(dataset[0])
                aepe_sum = torch.zeros([1, 1, demo_sample["target1"].shape[-2], demo_sample["target1"].shape[-1]]).cuda()
                flow_sum = torch.zeros([1, 1, demo_sample["target1"].shape[-2], demo_sample["target1"].shape[-1]]).cuda()

                for i in range(len(dataset)):
                    sample = dataset[i]
                    pred_flow = model(transformer(sample_to_torch_batch(sample)))["flow"]

                    aepe_sum += _upsample2d_as(_elementwise_epe(pred_flow, sample["target1"]), aepe_sum)
                    flow_sum += _upsample2d_as(torch.norm(sample["target1"], p=2, dim=1, keepdim=True), flow_sum)

                aepe_sum /= len(dataset)
                aepe_sum = aepe_sum.squeeze().detach().cpu().numpy()
                np.save(model_spatialAEPE_dir + "spatial_aepe.npy", aepe_sum, allow_pickle=False)

                flow_sum /= len(dataset)
                flow_sum = flow_sum.squeeze().detach().cpu().numpy()
                np.save(model_spatialAEPE_dir + "spatial_flow.npy", flow_sum, allow_pickle=False)
            else:
                aepe_sum = np.load(model_spatialAEPE_dir + "spatial_aepe.npy")
                flow_sum = np.load(model_spatialAEPE_dir + "spatial_flow.npy")

            if produce_base_images:
                grid = AGrid((1, 1), (aepe_sum.shape[-2], aepe_sum.shape[-1]), text_height=0, title_height=20)
                grid.place(0, 0, aepe_sum, colormap="hot")
                grid.title(model_name + " @ " + dataset_name)
                grid.save_image(model_spatialAEPE_dir + "spatial_aepe_" + dataset_name + "_" + model_name + ".png")

                grid = AGrid((1, 1), (flow_sum.shape[-2], flow_sum.shape[-1]), text_height=0, title_height=20)
                grid.place(0, 0, flow_sum, colormap="hot")
                grid.title(model_name + " @ " + dataset_name)
                grid.save_image(model_spatialAEPE_dir + "flow_magnitude_" + dataset_name + "_" + model_name + ".png")

            if produce_border_distance_images:
                aepe_edge_distances = compute_average_edge_distance_value(aepe_sum)
                flow_edge_distances = compute_average_edge_distance_value(flow_sum)
                raepe = 100*aepe_edge_distances/flow_edge_distances
                plt.figure()
                plt.title(model_name + " @ " + dataset_name)
                plt.plot(aepe_edge_distances, color="black")
                plt.plot(flow_edge_distances, color="red")
                plt.plot(raepe, color="black", linestyle="--")

                plt.ylim(0, max(min(max(np.nan_to_num(raepe)), 100), max(aepe_edge_distances), max(flow_edge_distances))*1.1)
                plt.tight_layout()
                #plt.show()
                plt.savefig(model_spatialAEPE_dir + "border_distance.png")
                plt.close()

            if produce_center_distance_images:
                pass

print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "finished")
