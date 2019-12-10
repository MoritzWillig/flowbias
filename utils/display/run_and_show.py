from collections import OrderedDict
import matplotlib.pyplot as plt
import torch

from models import PWCNet
from datasets import FlyingChairsFull
from utils.flow import flow_to_png

model_path = "/visinf/home/vimb01/projects/models/A_PWCNet-onChairs-20191121-171532/checkpoint_best.ckpt"

data_path = "/data/vimb01/FlyingChairs_sample402/FlyingChairs_sample402/data/"


def transform_state_dict(state_dict, transform):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_state_dict[transform(key)] = value
    return new_state_dict

model = PWCNet({})
stats = torch.load(model_path)
print("data keys:", stats.keys(), stats["state_dict"].keys())
# remove "_model" from name
model_params = transform_state_dict(stats["state_dict"], lambda name: name[7:])
model.load_state_dict(model_params)
model.eval().cuda()


dataset = FlyingChairsFull({}, data_path, photometric_augmentations=False)

def prepare_sample(sample):
    for key, value in sample.items():
        if key.startswith("input") or key.startswith("target"):
            sample[key] = value.unsqueeze(0).cuda()
    return sample

batch = prepare_sample(dataset[0])

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
