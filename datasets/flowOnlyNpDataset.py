import os
from glob import glob

import torch
import numpy as np
import torch.utils.data as data

from . import common


class FlowOnlyNpDataset(data.Dataset):

    def __init__(self, args, root):
        self._args = args

        flow_filenames = sorted(glob(os.path.join(root, "*.npy")))
        assert(len(flow_filenames) > 0)
        num_flows = len(flow_filenames)

        list_of_indices = range(num_flows)

        # ----------------------------------------------------------
        # Save list of actual filenames for flows
        # ----------------------------------------------------------
        self._flow_list = []
        for i in list_of_indices:
            flo = flow_filenames[i]
            self._flow_list += [flo]
        self._size = len(self._flow_list)

    def __getitem__(self, index):
        flo_filename = self._flow_list[index]

        # we assume the data is in the right format (no transposing)
        flo = torch.from_numpy(np.load(flo_filename)).float()

        # example filename
        basename = os.path.basename(flo_filename)

        example_dict = {
            "target1": flo,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size
