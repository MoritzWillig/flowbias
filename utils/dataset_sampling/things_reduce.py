import os
import pathlib
from shutil import copyfile

from flowbias.datasets.flyingThings3D import FlyingThings3dCleanTrain

things_root = "/data/vimb01/FlyingThings3D_subset/"
things_sample_root = "/data/vimb01/FlyingThings3D_sample401_subset/"

things_sample_im = os.path.join(things_sample_root, "train/image_clean/left/")
things_sample_flow_future = os.path.join(things_sample_root, "train/flow/left/into_future/")
things_sample_flow_past = os.path.join(things_sample_root, "train/flow/left/into_past/")


#create sample dir path
pathlib.Path(things_sample_im).mkdir(parents=True, exist_ok=True)
pathlib.Path(things_sample_flow_future).mkdir(parents=True, exist_ok=True)
pathlib.Path(things_sample_flow_past).mkdir(parents=True, exist_ok=True)

# copy data
dataset = FlyingThings3dCleanTrain({}, things_root, reduce_every_nth=49)

for i in range(len(dataset._image_list)):
    if i % 10 == 0:
        print(f"copying {i+1} / {len(dataset._image_list)}")
    im = dataset._image_list[i]
    flow = dataset._flow_list[i]

    copyfile(im[0], os.path.join(things_sample_im, os.path.basename(im[0])))
    copyfile(im[1], os.path.join(things_sample_im, os.path.basename(im[1])))
    copyfile(flow[0], os.path.join(things_sample_flow_future, os.path.basename(flow[0])))
    copyfile(flow[1], os.path.join(things_sample_flow_past, os.path.basename(flow[1])))

print('done')
