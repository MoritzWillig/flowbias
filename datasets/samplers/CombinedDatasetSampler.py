import torch.utils.data
import numpy as np


class CombinedDatasetBatchSampler(torch.utils.data.Sampler):

    def __init__(self, iterations, dataset_sizes, batch_size, dataset_sequence_mode="cycle"):
        """
        :param iterations:
        :param dataset_sizes:
        :param batch_size:
        :param dataset_sequence_mode: any of "cycle", "weighted_random"

        'cycle': switch the dataset for each batch (dataset indices itself is randomized)
        'weighted_random': draws batches from the datasets so that every sample has the same percentage of being drawn
        'equal_bias': draws batches from the datasets so that the dataset induced bias is equal for every dataset TODO
        """
        super(CombinedDatasetBatchSampler, self).__init__(dataset_sizes)
        self._iterations = iterations
        self._dataset_sizes = list(dataset_sizes)
        self._batch_sizes = [batch_size]*len(dataset_sizes) if isinstance(batch_size, int) else batch_size
        self._dataset_sequence_mode = dataset_sequence_mode

        if self._dataset_sequence_mode == "cycle":
            self._current_dataset = 0
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
        batch_size = self._batch_sizes[dataset_idx]

        if dataset_idx + batch_size <= dataset_size:
            self._dataset_indices[dataset_id] += batch_size
            return self._dataset_sequences[dataset_id][dataset_idx:dataset_idx+batch_size]
        else:
            # take remaining indices
            batch = self._dataset_sequences[dataset_id][dataset_idx:]
            # reshuffle indices
            sequence = np.random.permutation(dataset_size)
            self._dataset_sequences[dataset_id] = sequence
            # fill up batch from new permutation
            remaining = batch_size - (dataset_size - dataset_idx) - 1
            batch.extend(sequence[:remaining])
            self._dataset_indices[dataset_id] = remaining
            return batch

    def _generate_batch(self):
        dataset_id = None
        batch = None
        if self._dataset_sequence_mode == "cycle":
            dataset_id = self._current_dataset
            batch = self._batch_from_dataset(dataset_id)
            self._current_dataset = (self._current_dataset + 1) % len(self._dataset_sizes)
        elif self._dataset_sequence_mode == "weighted_random":
            dataset_id = np.random.choice(len(self._dataset_sizes), p=self._dataset_probabilities)
            batch = self._batch_from_dataset(dataset_id)
        elif self._dataset_sequence_mode == "equal_bias":
            raise NotImplementedError()

        return zip([dataset_id]*len(batch), batch)

    def __iter__(self):
        for idx in range(self._iterations):
            yield self._generate_batch()


class CTSKTrainDatasetBatchSampler(CombinedDatasetBatchSampler):

    def __init__(self, batch_size, iterations=None, dataset_sequence_mode="cycle"):
        dataset_sizes = [22232, 19635, 908, 160]
        if iterations is None:
            if dataset_sequence_mode == "cycle":
                # choose iteration number, so that the largest dataset is iterated once
                iterations = max(dataset_sizes)*batch_size
            else:
                raise ValueError("For a dataset_sequence_mode, other than 'cycle', the number of iterations must be passed")

        super().__init__(iterations, dataset_sizes, batch_size, dataset_sequence_mode)
