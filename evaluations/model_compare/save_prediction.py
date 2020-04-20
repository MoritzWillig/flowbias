import os
from datetime import datetime
import torch
import numpy as np

from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch

predFlow_dir = "/data/dataB/temp/predictedFlows/"
model_names = [
    "pwc_chairs", "I", "H",
    "pwc_kitti", "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti",
    "pwcWOX1_on_CTSK"
    ]
#model_name = "expertWOX1_CTSK_add01_expert3"


with torch.no_grad():
    for model_name in model_names:
        model, transformer = load_model_from_meta(model_name)
        model.cuda().eval()

        datasets = get_available_datasets(select_by_any_tag=["valid"], force_mode="test")
        print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), f"evaluating {model_name}")

        for dataset_name, dataset in datasets.items():
            print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), f"dataset {dataset_name} ({len(dataset)} items)")
            model_predFlow_dir = predFlow_dir + model_name + "_" + dataset_name + "/"
            os.makedirs(model_predFlow_dir, exist_ok=True)

            for i in range(len(dataset)):
                sample = dataset[i]
                pred_flow = model(transformer(sample_to_torch_batch(sample)))["flow"].squeeze().detach().cpu().numpy()

                predFlow_file = model_predFlow_dir + str(i) + ".npy"
                np.save(predFlow_file, pred_flow, allow_pickle=False)

print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "finished")
