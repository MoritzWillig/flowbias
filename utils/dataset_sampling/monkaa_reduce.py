import os


import pathlib
from shutil import copyfile

from flowbias.datasets.monkaa import MonkaaFinalTrain

monkaa_root = "/data/vimb01/monkaa_finalpass/"
monkaa_sample_root = "/data/vimb01/monkaa_finalpass_sample411/"

full_dataset = MonkaaFinalTrain({}, monkaa_root)
print(f"full dataset size: {len(full_dataset)}")


# copy data
dataset = MonkaaFinalTrain({}, monkaa_root, reduce_every_nth=20)
print(f"reduced dataset size: {len(dataset)}")

for i in range(len(dataset._image_list)):
    if i % 10 == 0:
        print(f"copying {i+1} / {len(dataset._image_list)}")
    im_0 = dataset._image_list[i][0]
    im_1 = dataset._image_list[i][1]
    flow = dataset._flow_list[i]

    im0_sub = f"{monkaa_sample_root}{im_0[len(monkaa_root):]}"
    im1_sub = f"{monkaa_sample_root}{im_1[len(monkaa_root):]}"
    flow_sub = f"{monkaa_sample_root}{flow[len(monkaa_root):]}"

    # create directory structure
    pathlib.Path(os.path.dirname(im0_sub)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(flow_sub)).mkdir(parents=True, exist_ok=True)

    copyfile(im_0, im0_sub)
    copyfile(im_1, im1_sub)
    copyfile(flow, flow_sub)

print('done')
