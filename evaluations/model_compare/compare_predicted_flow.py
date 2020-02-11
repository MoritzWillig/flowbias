import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from flowbias.utils.flow import compute_color
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta
from flowbias.utils.model_loading import sample_to_torch_batch

dataset_name = "sintelFinalValid" #flyingChairsValid flyingThingsCleanValid sintelFinalValid kitti2015Valid
sample_id = 130  # 130 29 67

#models = ["A", "I", "H", "W"] # base models
#models = ["A", "F", "pwcWOX1_chairs", "expert_split02_expert0"] # chairs_base, chairs_fine, WOX1, expert_chairs
models = [
    "expert_split02_expert0", "expert_split02_expert1", "expert_split02_expert2", "expert_split02_expert3",
    "expert_add01_expert0", "expert_add01_expert1", "expert_add01_expert2", "expert_add01_expert3",
    "A", "pwc_chairs_iter_148", "I", "F",
    "expert_add01_no_expert", "pwc_on_CTSK"
]
#models = ["W", "pwcWOX1_kitti", "A", "S", "WOX1Blind_ks", "WOX1Blind_sk"]  # kitti networks

num_cols = 4

datasets = get_available_datasets(restrict_to=[dataset_name], force_mode="test")
dataset = datasets[dataset_name]
sample = sample_to_torch_batch(dataset[sample_id])
print(sample.keys())

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


class AGrid:

    def __init__(self, grid, image_shape):
        self._im_shape = list(image_shape).copy()
        self._a = np.zeros((grid[1]*image_shape[0], grid[0]*image_shape[1], 3))

    def place(self, x, y, image):
        self._a[
            y*self._im_shape[0]:y*self._im_shape[0] + image.shape[0],
            x*self._im_shape[1]:x*self._im_shape[1] + image.shape[1],
            :
        ] = image

    def get_image(self):
        return self._a


plt_size = (num_cols, 1+int(math.ceil(len(models)/num_cols)))
grid = AGrid(plt_size, gt_flow.shape)

grid.place(0, 0, im1)
grid.place(1, 0, im2)
grid.place(2, 0, gt_flow)

for i, result in enumerate(results):
    flow = result["flow"].cpu().detach().numpy()
    #grid.place(i, 1, compute_color(flow[0, 0, :,:], flow[0, 1, :,:])/255)
    grid.place(i % num_cols, 1+(i//num_cols), compute_color(flow[0, 0, :,:], flow[0, 1, :,:])/255)


fig = plt.figure()
plt.imshow(grid.get_image())
plt.tight_layout()
plt.show()


