from collections import OrderedDict
import matplotlib.pyplot as plt
import torch

from flowbias.models import PWCNet
from flowbias.datasets import FlyingChairsFull
from flowbias.utils.eval.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.utils.flow import flow_to_png

checkpoint_path = "/visinf/home/vimb01/projects/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt"
data_path = "/data/vimb01/FlyingChairs_sample402/FlyingChairs_sample402/data/"


model = PWCNet({})
load_model_parameters(model, checkpoint_path)
model.eval().cuda()


dataset = FlyingChairsFull({}, data_path, photometric_augmentations=False)
batch = sample_to_torch_batch(dataset[0])
results = model(batch)["flow"].detach().cpu().numpy()


fig = plt.figure()

ax = fig.add_subplot(2, 2, 1)
ax.title.set_text(f"im1")
plt.imshow(dataset[0]["input1"].detach().cpu().numpy().transpose([1,2,0]))

ax = fig.add_subplot(2, 2, 2)
ax.title.set_text(f"im2")
plt.imshow(dataset[0]["input1"].detach().cpu().numpy().transpose([1,2,0]))

ax = fig.add_subplot(2, 2, 3)
ax.title.set_text(f"predicted")
plt.imshow(flow_to_png(results[0,:,:,:]))

ax = fig.add_subplot(2, 2, 4)
ax.title.set_text(f"target")
plt.imshow(flow_to_png(dataset[0]["target1"].detach().cpu().numpy()))

plt.tight_layout()
plt.show()
