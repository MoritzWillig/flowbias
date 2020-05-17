import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import torch
import torch.nn as nn
import numpy as np

from flowbias.models import PWCLinCombAppliedConnector

correlation_path = "/visinf/home/vimb01/projects/evals/results_A_B"
resultingModelPath = "/visinf/home/vimb01/projects/evals/results_A_B/connector.pt"

num_levels = 5

# computes a weighted linear combination
# of the features A to match the B features.

# features are weighted according to their correlation
# mean and std_dev are adjusted


weights_full = []
means_full = []
for l in range(num_levels):
    corr = np.load(f"{correlation_path}/{l}_corr.npy")
    mean = np.load(f"{correlation_path}/{l}_mean.npy")
    std_dev = np.load(f"{correlation_path}/{l}_std_dev.npy")

    # extract x1 features (remove correlation and flow features)
    h = corr.shape[0] // 2
    corr = corr[h+81:-2, 81:h-2]
    mean = np.hstack((mean[81:h-2], mean[h+81:-2]))
    std_dev = np.hstack((std_dev[81:h-2], std_dev[h+81:-2]))

    num_features = corr.shape[0]
    print(">>", num_features)
    summed_weight = np.sum(corr, axis=1)

    weights = np.zeros((num_features, num_features))
    bias = np.zeros((num_features, num_features))
    for f in range(num_features):
        # adjust std_dev
        weights[f, :] =\
            (corr[f, :] / summed_weight) * (std_dev[num_features + f] / std_dev[:num_features])
        # shift mean
        bias[f, :] = -mean[:num_features] + mean[num_features+f]

    means_full.append(bias)
    weights_full.append(weights)


connector = PWCLinCombAppliedConnector({})
for i in range(num_levels):
    print(means_full[i].shape, weights_full[i].shape, connector.convs[i].weight.size(), connector.shifts[i].size())

    connector.shifts[i].data[:] = torch.tensor(means_full[i][:])
    connector.convs[i].weight[:, :, :, :] = torch.tensor(weights_full[i][:, :, None, None])
torch.save(connector.state_dict(), resultingModelPath)
