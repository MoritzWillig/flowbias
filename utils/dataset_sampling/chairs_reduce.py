import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import pathlib
from shutil import copyfile

from datasets.flyingchairs import FlyingChairsTrain

chairs_root = "/data/vimb01/FlyingChairs_release/data/"
chairs_sample_root = "/data/vimb01/FlyingChairs_sample402/data/"

#create sample dir path
pathlib.Path(chairs_sample_root).mkdir(parents=True, exist_ok=True)

# copy data
dataset = FlyingChairsTrain({}, chairs_root, reduce_every_nth=56)

for i in range(len(dataset._image_list)):
    if i % 10 == 0:
        print(f"copying {i+1} / {len(dataset._image_list)}")
    im = dataset._image_list[i]
    flow = dataset._flow_list[i]

    copyfile(im[0], os.path.join(chairs_sample_root, os.path.basename(im[0])))
    copyfile(im[1], os.path.join(chairs_sample_root, os.path.basename(im[1])))
    copyfile(flow, os.path.join(chairs_sample_root, os.path.basename(flow)))

print('done')
