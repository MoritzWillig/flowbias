from __future__ import absolute_import, division, print_function

import torch.utils.data as data
from glob import glob

from datasets import common
from evaluations.horizontal_stack.tools import load_sample_level


class PWCInterfaceDataset(data.Dataset):
    def __init__(self,
                 args,
                 rootA,
                 rootB,
                 level,
                 reduce_every_nth=None,
                 inverse_reduce=False):

        self._args = args
        self.level = level
        self._reduce_every_nth = reduce_every_nth
        self._inverse_reduce = inverse_reduce

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

        sA_filename = self._filenamesA[index]
        sB_filename = self._filenamesB[index]

        out_corr_reluA, x1A, flowA, lA = load_sample_level(sA_filename[index], self.level)
        out_corr_reluB, x1B, flowB, lB = load_sample_level(sB_filename[index], self.level)

        # convert flow to FloatTensor
        out_corr_reluA = common.numpy2torch(out_corr_reluA)
        x1A = common.numpy2torch(x1A)
        flowA = common.numpy2torch(flowA)
        #lA = common.numpy2torch(lA)

        out_corr_reluB = common.numpy2torch(out_corr_reluB)
        x1B = common.numpy2torch(x1B)
        flowB = common.numpy2torch(flowB)
        #lB = common.numpy2torch(lB)

        example_dict = {
            "input_out_corr_relu": out_corr_reluA,
            "input_x1": x1A,
            "input_flow": flowA,
            "target_out_corr_relu": out_corr_reluB,
            "target_x1": x1B,
            "target_flow": flowB,
            "index": index
        }

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
            reduce_every_nth=10,
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
            reduce_every_nth=10,
            inverse_reduce=True)
