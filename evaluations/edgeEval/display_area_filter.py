import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from flowbias.config import Config
from flowbias.utils.flow import compute_color
from flowbias.utils.meta_infrastructure import get_available_datasets
from flowbias.evaluations.edgeEval.area_filter import AreaFilter
from flowbias.utils.model_loading import sample_to_torch_batch

dataset_name = "sintelFinalValid" #"flyingChairsValid"
datasets = get_available_datasets(restrict_to=[dataset_name])
dataset = datasets[dataset_name]

#sample_id = 5
sample_id = 712
sample = sample_to_torch_batch(dataset[sample_id])
print(sample.keys())
flow = sample["target1"]

#area_filter = AreaFilter([1,2,4,8], channels=2)
#factor 1.6 to resemble a laplacian of gaussian
#area_filter = AreaFilter([1, 1.6, 2.56, 4.09, 6.55, 10.48], channels=2)
area_filter = AreaFilter([1, 1.6, 2.56, 4.09, 6.55], channels=2)


#normalize flow
flow_n = flow / torch.norm(flow, dim=1, keepdim=True)
dog = area_filter.compute_dog(flow_n).cpu().numpy()
print(dog.shape)



isum = np.sum(np.abs(dog)[0,:,:,:], axis=0)
mp = area_filter._max_pad

im1 = sample["input1"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])[mp:-mp,mp:-mp,:]
im2 = sample["input2"].cpu().numpy()[0,:,:,:].transpose([1, 2, 0])[mp:-mp,mp:-mp,:]
flow = flow.cpu().numpy()[0,:,:,:].transpose([1, 2, 0])[mp:-mp,mp:-mp,:]
flow = compute_color(flow[:,:,0], flow[:,:,1])

plt.figure()
plt.subplot(2,2,1)
plt.imshow(im1)
plt.subplot(2,2,2)
plt.imshow(im2)
plt.subplot(2,2,3)
plt.imshow(flow)
plt.subplot(2,2,4)
plt.imshow(isum, cmap="gray")
plt.show()


print(np.min(isum), np.max(isum))
isum /= np.max(isum)


im = Image.fromarray(isum*255)
if im.mode != 'RGB':
    im = im.convert('RGB')
im.save(f"{Config.temp_directory}/edges/flow_dog_{dataset_name}_{sample_id}.png")



