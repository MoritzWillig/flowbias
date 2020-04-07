import torch.utils.data
import numpy as np


class CombinedDatasetBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset_sizes, batch_size, dataset_sequence_mode="cycle"):
        """
        :param iterations:
        :param dataset_sizes:
        :param batch_size:
        :param dataset_sequence_mode: any of "cycle", "weighted_random"

        'cycle': switch the dataset for each batch (dataset indices itself is randomized)
        'mixed': creates a batch out of samples from every dataset
        'weighted_random': draws batches from the datasets so that every sample has the same percentage of being drawn
        'equal_bias': draws batches from the datasets so that the dataset induced bias is equal for every dataset TODO
        """
        super(CombinedDatasetBatchSampler, self).__init__(dataset_sizes)
        self._dataset_sizes = list(dataset_sizes)
        self._batch_sizes = [batch_size]*len(dataset_sizes) if isinstance(batch_size, int) else batch_size
        self._dataset_sequence_mode = dataset_sequence_mode

        if self._dataset_sequence_mode == "cycle":
            self._current_dataset = 0
        elif self._dataset_sequence_mode == "mixed":
            # same batch size for all datasets
            assert(all([bs == self._batch_sizes[0] for bs in self._batch_sizes]))
            # make sure every we can draw the same number of samples from every dataset
            assert(self._batch_sizes[0] % len(self._dataset_sizes) == 0)
            self._num_dataset_samples = self._batch_sizes[0] // len(self._dataset_sizes)
            self._batch_sizes = [self._num_dataset_samples for _ in self._batch_sizes]
        elif self._dataset_sequence_mode == "weighted_random":
            total_samples = sum(self._dataset_sizes)
            self._dataset_probabilities = [dataset_size / total_samples for dataset_size in self._dataset_sizes]
        elif self._dataset_sequence_mode == "equal_bias":
            raise NotImplementedError()
        else:
            raise ValueError("dataset_sequence_mode has to be 'cycle' or 'weighted_random'")

        self._dataset_indices = [0 for i in range(len(self._dataset_sizes))]
        self._dataset_sequences = [np.random.permutation(n) for n in self._dataset_sizes]

    def _batch_from_dataset(self, dataset_id):
        dataset_size = self._dataset_sizes[dataset_id]
        dataset_idx = self._dataset_indices[dataset_id]
        batch_size = self._batch_sizes[dataset_id]

        if dataset_idx + batch_size <= dataset_size:
            self._dataset_indices[dataset_id] += batch_size
            batch = self._dataset_sequences[dataset_id][dataset_idx:dataset_idx+batch_size]
        else:
            # take remaining indices
            batch = np.empty(batch_size, dtype=np.int)
            idcs_from_old_perm = (dataset_size - dataset_idx)
            batch[:idcs_from_old_perm] = self._dataset_sequences[dataset_id][dataset_idx:]
            # reshuffle indices
            sequence = np.random.permutation(dataset_size)
            self._dataset_sequences[dataset_id] = sequence
            # fill up batch from new permutation
            remaining = batch_size - idcs_from_old_perm
            batch[:remaining] = sequence[:remaining]
            self._dataset_indices[dataset_id] = remaining
        return batch

    def _generate_batch(self):
        if self._dataset_sequence_mode == "cycle":
            dataset_id = self._current_dataset
            batch = self._batch_from_dataset(dataset_id)
            self._current_dataset = (self._current_dataset + 1) % len(self._dataset_sizes)
            return list(zip([dataset_id] * len(batch), batch))
        elif self._dataset_sequence_mode == "mixed":
            batch = []
            for dataset_id in range(len(self._dataset_sizes)):
                dataset_samples = self._batch_from_dataset(dataset_id)
                batch.extend(list(zip([dataset_id]*self._batch_sizes[dataset_id], dataset_samples)))
            return batch
        elif self._dataset_sequence_mode == "weighted_random":
            dataset_id = np.random.choice(len(self._dataset_sizes), p=self._dataset_probabilities)
            batch = self._batch_from_dataset(dataset_id)
            return list(zip([dataset_id] * len(batch), batch))
        elif self._dataset_sequence_mode == "equal_bias":
            raise NotImplementedError()
        else:
            raise RuntimeError("unknown dataset sequence mode")

    def __iter__(self):
        while True:
            batch = self._generate_batch()
            yield batch


class CTSKTrainDatasetBatchSampler(CombinedDatasetBatchSampler):
    """
    Sampler for a combined training of Chairs, Things, Sintel and KITTI.
    """

    def __init__(self, batch_size, dataset_sequence_mode="cycle"):
        dataset_sizes = [22232, 19635, 908, 160]
        super().__init__(dataset_sizes, batch_size, dataset_sequence_mode)


class CTSTrainDatasetBatchSampler(CombinedDatasetBatchSampler):
    """
    Sampler for a combined training of Chairs, Things and Sintel
    """

    def __init__(self, batch_size, dataset_sequence_mode="cycle"):
        dataset_sizes = [22232, 19635, 908]
        super().__init__(dataset_sizes, batch_size, dataset_sequence_mode)
