
class CombinedDataset:

    def __init__(self, datasets):
        self._datasets = list(datasets)
        self._size = sum([len(dataset) for dataset in self._datasets])

    def __getitem__(self, index):
        index = index % self._size

        ...

        example_dict = {
            "input1": im1,
            "input2": im2,
            "target1": flo,
            "target_occ1": target_occ,
            "index": index,
            "basename": basename
        }

        return example_dict

    def __len__(self):
        return self._size
