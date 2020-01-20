import torchviz
import matplotlib.pyplot as plt

from flowbias.utils.eval.model_loading import prepare_sample
from flowbias.models import PWCNet
from flowbias.datasets import FlyingThings3dCleanTrain

model = PWCNet({})
dataset = FlyingThings3dCleanTrain({}, "/home/moritz/projects/datasets/MPI-Sintel-complete", photometric_augmentations=False)
params=None

y = model(prepare_sample(dataset[0]), params)

dot = torchviz.make_dot(y)

print(dot)
#plt.imshow(dot)
#plt.show()
