from __future__ import absolute_import, division, print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common


class SubsampledDataset(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 dstype="full"):

        # subsampled datasets are only allows as "full"
        assert(dstype == "full")

        self._args = args

        # -------------------------------------------------------------
        # filenames for all input images and target flows
        # -------------------------------------------------------------
        image_filenames = sorted(glob(os.path.join(root, "*.png")))
        flow_filenames = sorted(glob(os.path.join(root, "*.flo")))
        assert (len(image_filenames) > 0)
        assert(len(image_filenames)/2 == len(flow_filenames))
        num_flows = len(flow_filenames)

        list_of_indices = range(num_flows)

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []
        for i in list_of_indices:
            im1 = image_filenames[2 * i]
            im2 = image_filenames[2 * i + 1]
            flo = flow_filenames[i]
            self._image_list += [[im1, im2]]
            self._flow_list += [flo]
        self._size = len(self._image_list)
        assert(len(self._image_list) == len(self._flow_list))

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)
        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def __getitem__(self, index):
        index = index % self._size

        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]
        flo_filename = self._flow_list[index]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_np0 = common.read_flo_as_float32(flo_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # convert flow to FloatTensor
        flo = common.numpy2torch(flo_np0)

        # example filename
        basename = os.path.basename(im1_filename)

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size