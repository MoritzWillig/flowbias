from __future__ import absolute_import, division, print_function

import torch.utils.data as data
from glob import glob

from flowbias.datasets import common
from flowbias.evaluations.horizontal_stack.tools import load_sample


class PWCInterfaceDataset(data.Dataset):
    def __init__(self,
                 args,
                 rootA,
                 rootB,
                 reduce_every_nth=None,
                 inverse_reduce=False):

        self._args = args
        self._reduce_every_nth = reduce_every_nth
        self._inverse_reduce = inverse_reduce
        self.num_levels = 5

        # -------------------------------------------------------------
        # filenames for all input images and target flows
        # -------------------------------------------------------------
        filenamesA = sorted(glob(rootA))
        filenamesB = sorted(glob(rootB))
        self._filenamesA = []
        self._filenamesB = []

        for i in range(len(filenamesA)):
            if self._reduce_every_nth is not None:
                if (i % self._reduce_every_nth != 0) == (not self._inverse_reduce):
                    continue
            self._filenamesA.append(filenamesA[i])
            self._filenamesB.append(filenamesB[i])

        assert(len(self._filenamesA) == len(self._filenamesB))
        self._size = len(self._filenamesA)

    def __getitem__(self, index):
        index = index % self._size

        example_dict = {
            "index": index
        }

        sA_filename = self._filenamesA[index]
        sB_filename = self._filenamesB[index]

        out_corr_reluA, x1A, flowA, lA = load_sample(sA_filename[index])
        out_corr_reluB, x1B, flowB, lB = load_sample(sB_filename[index])

        for i in range(self.num_levels):
            # read level and convert flow to FloatTensor
            x1Al = common.numpy2torch(x1A[i])
            x1Bl = common.numpy2torch(x1B[i])

            example_dict[f"input_x1_{i}"] = x1Al
            example_dict[f"target_x1_{i}"] = x1Bl
        return example_dict

    def __len__(self):
        return self._size


class PWCInterfaceDatasetTrain(PWCInterfaceDataset):
    def __init__(self,
                 args,
                 rootA,
                 rootB):
        super(PWCInterfaceDataset, self).__init__(
            args,
            rootA=rootA,
            rootB=rootB,
            reduce_every_nth=20,
            inverse_reduce=False)


class PWCInterfaceDatasetValid(PWCInterfaceDataset):
    def __init__(self,
                 args,
                 rootA,
                 rootB):
        super(PWCInterfaceDataset, self).__init__(
            args,
            rootA=rootA,
            rootB=rootB,
            reduce_every_nth=20,
            inverse_reduce=True)
