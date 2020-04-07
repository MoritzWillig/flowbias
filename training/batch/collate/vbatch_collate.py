from torch.utils.data.dataloader import default_collate


class VBatchGroupingCollator:
    """
    groups samples into vbatches by vbatch_key.
    """

    def __init__(self, vbatch_key):
        self._vbatch_key = vbatch_key

    def collate_samples(self, samples):
        # sort-in samples
        vbatches = {}
        for sample in samples:
            vbatch_key = sample[self._vbatch_key]
            if vbatch_key not in vbatches:
                vbatches[vbatch_key] = []
            vbatches[vbatch_key].append(sample)

        batch = {
            "virtual_batch": True,
            "virtual_batches": [default_collate(samples) for samples in vbatches.values()]
        }
        return batch

    def __call__(self, samples, *args, **kwargs):
        return self.collate_samples(samples)


class CombinedDatasetVBatchCollator(VBatchGroupingCollator):
    """
    groups samples by dataset
    """

    def __init__(self):
        super().__init__("dataset")
