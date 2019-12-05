from __future__ import absolute_import, division, print_function

import os
from glob import glob
import re
import torch.utils.data as data

from torchvision import transforms as vision_transforms

from . import transforms
from . import common

import numpy as np


def fillingInNaN(flow):
    h, w, c = flow.shape
    indices = np.argwhere(np.isnan(flow))
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


class Monkaa(data.Dataset):
    TEST_IS_EVERY_NTH = 20

    def __init__(self,
                 args,
                 images_root,
                 flow_root,
                 reduce_every_nth=None,
                 side="left",
                 set="train",
                 photometric_augmentations=False):

        # flow: <scene_name>/into_<future|past>/<left|right>/<filename>.pfm
        # images: funnyworld_camera2_augmented1_x2/right/0346.png

        self._args = args
        self._reduce_every_nth = reduce_every_nth
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")
        if not os.path.isdir(flow_root):
            raise ValueError("Flow directory '%s' not found!")

        flow_f_filenames = sorted(glob(os.path.join(flow_root, f"*/into_future/{side}/*.pfm")))

        self._image_list = []
        self._flow_list = []

        assert len(flow_f_filenames) != 0

        side_char = "L" if side == "left" else "R"

        # ----------------------------------------------------------
        # Save list of actual filenames for inputs and flows
        # ----------------------------------------------------------
        abs_idx = 0
        num = 0  # index in the current dataset (train or test)
        for ii in range(len(flow_f_filenames)):

            flo_f = flow_f_filenames[ii]

            idx_f = re.search(r"\d+", os.path.basename(flo_f)).group(0)
            idx_b = str(int(idx_f) + 1).zfill(len(idx_f))

            scene_name = os.path.dirname(os.path.dirname(os.path.dirname(flo_f)))

            im1 = os.path.join(images_root, scene_name, side, f"{idx_f}_{side_char}.png")
            im2 = os.path.join(images_root, scene_name, side, f"{idx_b}_{side_char}.png")

            if not os.path.isfile(flo_f) or not os.path.isfile(im1) or not os.path.isfile(im2):
                continue

            # test if the index belongs to the current dataset split (train or valid)
            is_test = (abs_idx % Monkaa.TEST_IS_EVERY_NTH) == 0
            if (set == "train") == is_test:
                abs_idx += 1
                continue

            # test if we skip the current the index (useful for splitting up the training set)
            if (self._reduce_every_nth is not None) and (num % self._reduce_every_nth != 0):
                num += 1
                abs_idx += 1
                continue

            num += 1
            abs_idx += 1
            self._image_list += [[im1, im2]]
            self._flow_list += [flo_f]


        self._size = len(self._image_list)

        assert len(self._image_list) == len(self._flow_list)
        assert len(self._image_list) != 0

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
        #flo_f_filename = self._flow_list[index][0]
        #flo_b_filename = self._flow_list[index][1]

        # read float32 images and flow
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)
        flo_f_np0 = common.read_flo_as_float32(flo_f_filename)
        #flo_b_np0 = common.read_flo_as_float32(flo_b_filename)

        # temp - check isnan
        if np.any(np.isnan(flo_f_np0)):
            flo_f_np0 = fillingInNaN(flo_f_np0)

        #if np.any(np.isnan(flo_b_np0)):
        #    flo_b_np0 = fillingInNaN(flo_b_np0)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # convert flow to FloatTensor
        flo_f = common.numpy2torch(flo_f_np0)
        #flo_b = common.numpy2torch(flo_b_np0)

        # example filename
        basename = os.path.basename(im1_filename)[:5]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo_f,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size


class MonkaaFinalTrain(Monkaa):
    def __init__(self,
                 args,
                 root,
                 reduce_every_nth=None,
                 photometric_augmentations=True):
        images_root = os.path.join(root, "frames_finalpass")
        flow_root = os.path.join(root, "optical_flow")
        super(MonkaaFinalTrain, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            reduce_every_nth=reduce_every_nth,
            set="train",
            photometric_augmentations=photometric_augmentations)


class MonkaaFinalValid(Monkaa):
    def __init__(self,
                 args,
                 root,
                 reduce_every_nth=None,
                 photometric_augmentations=False):
        images_root = os.path.join(root, "frames_finalpass")
        flow_root = os.path.join(root, "optical_flow")
        super(MonkaaFinalValid, self).__init__(
            args,
            images_root=images_root,
            flow_root=flow_root,
            reduce_every_nth=reduce_every_nth,
            set="valid",
            photometric_augmentations=photometric_augmentations)
