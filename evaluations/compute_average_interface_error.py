from flowbias.datasets.pwcInterfaceDataset import PWCInterfaceDatasetValid
from flowbias.losses import MSEConnectorLoss
from flowbias.models import PWCTrainableConvConnector33
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
import torch
import numpy as np

"""
Computes the error of an interface, when always predicting the average value (= equals stddev).
"""

class ValArgs:
    def __init__(self):
        self.batch_size = 1


connectors = ["a", "i", "h", "w"]

interfaces_roots = {
    "ac": "/data/dataA/model_interfaces/A_chairs",
    "ak": "/data/dataA/model_interfaces/A_kitti",
    "as": "/data/dataA/model_interfaces/A_sintel",
    "at": "/data/dataA/model_interfaces/A_things",
    "hc": "/data/dataA/model_interfaces/H_chairs",
    "hk": "/data/dataA/model_interfaces/H_kitti",
    "hs": "/data/dataA/model_interfaces/H_sintel",
    "ht": "/data/dataA/model_interfaces/H_things",
    "ic": "/data/dataA/model_interfaces/I_chairs",
    "ik": "/data/dataA/model_interfaces/I_kitti",
    "is": "/data/dataA/model_interfaces/I_sintel",
    "it": "/data/dataA/model_interfaces/I_things",
    "wc": "/data/dataA/model_interfaces/W_chairs",
    "wk": "/data/dataA/model_interfaces/W_kitti",
    "ws": "/data/dataA/model_interfaces/W_sintel",
    "wt": "/data/dataA/model_interfaces/W_things"
}

datasets = ["c", "k", "s", "t"]


for connector in connectors:
    for dataset_char in datasets:
        rootA = interfaces_roots[connector+dataset_char]
        dataset = PWCInterfaceDatasetValid(ValArgs(), rootA, rootA)

        demo_sample = dataset[0]

        # compute mean
        means = [np.zeros((len(dataset), demo_sample[f"target_x1_{i}"].size()[0])) for i in range(5)]
        for j in range(len(dataset)):
            sample = dataset[j]
            for i in range(5):
                means[i][j, :] = torch.mean(sample[f"target_x1_{i}"], (1, 2)).cpu().detach().numpy()
        for i in range(5):
            means[i] = np.mean(means[i], 0)
        # means now contains the mean of every feature, in every layer

        # compute variance
        mse = [np.zeros((len(dataset), demo_sample[f"target_x1_{i}"].size()[0])) for i in range(5)]
        for j in range(len(dataset)):
            sample = dataset[j]
            for i in range(5):
                mse[i][j, :] = np.mean((sample[f"target_x1_{i}"].cpu().detach().numpy() - means[i][:, None, None])**2, (1, 2))
        for i in range(5):
            mse[i] = np.mean(mse[i], 0)

        # weight stddevs by image size
        #sizes = np.array([demo_sample[f"target_x1_{i}"].size()[1] * demo_sample[f"target_x1_{i}"].size()[2] for i in range(5)])
        #sizes = sizes / np.sum(sizes)
        #
        #
        # the MSEConnectorLoss just averages the layer losses
        avg_mse = np.sum([np.mean(mse[i]) for i in range(5)]) / 5  # std dev is computed as a whole layer
        print(f"{connector}{dataset_char} mse: {avg_mse} | total_mse: {len(dataset) * avg_mse} | stdev: {np.sqrt(avg_mse)}")
        #for i in range(5):
        #    print(f"layer {i}", means[i].shape)
        #    print(f"layer {i}", stddevs[i].shape)

print("finished")
