import os
import pathlib
from shutil import copyfile

from flowbias.datasets import SubsampledDataset
from flowbias.datasets import SintelTrainingCleanTrain

sintel_root = "/data/dataB/datasets/MPI-Sintel-complete/"
sintel_subset_root = "/data/dataB/datasets/MPI-Sintel_subset400/"

#create sample dir path
pathlib.Path(sintel_subset_root).mkdir(parents=True, exist_ok=True)

# copy data
dataset = SintelTrainingCleanTrain({}, sintel_root, photometric_augmentations=False)

every_nth = 2.275
n = len(dataset._image_list) / every_nth

ct = 0
ii = 0
while ii < len(dataset._image_list):
    i = int(ii)
    if ct % 10 == 0:
        print(f"copying {ct+1} / {n}")
    im1_filename = dataset._image_list[i][0]
    im2_filename = dataset._image_list[i][1]
    flo_filename = dataset._flow_list[i]

    copyfile(im1_filename, os.path.join(sintel_subset_root, f"{ct}_0.png"))
    copyfile(im2_filename, os.path.join(sintel_subset_root, f"{ct}_1.png"))
    copyfile(flo_filename, os.path.join(sintel_subset_root, f"{ct}_flow.flo"))

    ct += 1
    ii += every_nth

print(f'{ct} done')

subdataset = SubsampledDataset({}, sintel_subset_root)
print(f"created subdataset with {len(subdataset)} samples")
