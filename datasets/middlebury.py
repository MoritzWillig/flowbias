from __future__ import absolute_import, division, print_function

import os
import pathlib

import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import numpy as np


def fillingInMask(flow, valid_mask):
    h, w, c = flow.shape
    indices = np.argwhere(~valid_mask)
    neighbors = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for ii, idx in enumerate(indices):
        sum_sample = 0
        count = 0
        for jj in range(0, len(neighbors) - 1):
            hh = idx[0] + neighbors[jj][0]
            ww = idx[1] + neighbors[jj][1]
            if hh < 0 or hh >= h:
                continue
            if ww < 0 or ww >= w:
                continue
            sample_flow = flow[hh, ww, idx[2]]
            if np.isnan(sample_flow):
                continue
            sum_sample += sample_flow
            count += 1
        if count is 0:
            print('FATAL ERROR: no sample')
        flow[idx[0], idx[1], idx[2]] = sum_sample / count

    return flow


class MiddleburyTrainValid(data.Dataset):
    """
    validation on training data.
    """

    def __init__(self,
                 args,
                 images_root,
                 flow_root,
                 photometric_augmentations=False):

        self._args = args

        if not os.path.isdir(images_root):
            raise ValueError(f"Image directory '{images_root}' not found!")
        if not os.path.isdir(flow_root):
            raise ValueError(f"Flow directory '{flow_root}' not found!")

        #all_img1_filenames = sorted(glob(os.path.join(images_root, "/*/*frame10.png")))
        #not all img files have a gt flow
        #all_img1_filenames = sorted(glob(images_root+"/other-color-allframes/other-data/*/frame10.png"))
        #all_img2_filenames = sorted(glob(images_root+"/other-color-allframes/other-data/*/frame11.png"))
        flow_filenames = sorted(glob(flow_root+"/other-gt-flow/*/flow10.flo"))
        assert len(flow_filenames) != 0

        all_img1_filenames = []
        all_img2_filenames = []
        for flow_filename in flow_filenames:
            dirname = pathlib.PurePath(flow_filename).parent.name
            all_img1_filenames.append(images_root+"/other-color-allframes/other-data/"+dirname+"/frame10.png")
            all_img2_filenames.append(images_root+"/other-color-allframes/other-data/"+dirname+"/frame11.png")
        assert len(all_img1_filenames) == len(all_img2_filenames)
        assert len(all_img1_filenames) == len(flow_filenames)


        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []

        for im1, im2, flow in zip(all_img1_filenames, all_img2_filenames, flow_filenames):
            self._image_list.append([im1, im2])
            self._flow_list.append(flow)

        self._size = len(self._image_list)

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
        flo_f_filename = self._flow_list[index]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_f_np0 = common.read_flo_as_float32(flo_f_filename)
        valid_mask = flo_f_np0 != 1666666800.0

        if np.any(valid_mask):
            flo_f_np0 = fillingInMask(flo_f_np0, valid_mask)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # convert flow to FloatTensor
        flo_f = common.numpy2torch(flo_f_np0)
        valid_mask_f = common.numpy2torch(valid_mask)

        # example filename
        basename = os.path.basename(im1_filename)[:6]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo_f,
            "target2": flo_f,
            "index": index,
            "basename": basename,
            "input_valid": valid_mask_f
        }
        return example_dict

    def __len__(self):
        return self._size
