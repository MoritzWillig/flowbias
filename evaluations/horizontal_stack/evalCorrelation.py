from glob import glob
import numpy as np
from flowbias.utils.data_loading import load_sample

sample_interface_pathA = "/data/vimb01/evaluations/A_onThings_interface/*"
sample_interface_pathB = "/data/vimb01/evaluations/C_onThings_interface/*"

resulting = "/data/vimb01/evaluations/corr_Athings_Cthings/"

all_sample_filenamesA = sorted(glob(sample_interface_pathA))
all_sample_filenamesB = sorted(glob(sample_interface_pathB))

assert(len(all_sample_filenamesA) == len(all_sample_filenamesB))

_, _, flow, _ = load_sample(all_sample_filenamesA[0])
num_levels = len(flow)
print(f"num_levels: {num_levels}")

def extractLevel(interface, level):
    out_corr_relu, x1, flow, l = interface
    out_corr_reluS = np.squeeze(out_corr_relu[level], axis=(0,))
    x1S = np.squeeze(x1[level], axis=(0, 1))
    flowS = np.squeeze(flow[level], axis=(0, 1))
    lS = l[level]
    return out_corr_reluS, x1S, flowS, lS

def flatten_merge(out_corr_relu, x1, flow):
    out_corr_relu = np.reshape(out_corr_relu, (out_corr_relu.shape[0], -1))
    x1 = np.reshape(x1, (x1.shape[0], -1))
    flow = np.reshape(flow, (flow.shape[0], -1))
    return np.vstack((out_corr_relu, x1, flow))

def custom_coeff(data):
    #formula: en.wikipedia.org/wiki/Correlation_and_dependence#Sample_correlation_coefficient
    num_rows = data.shape[0]
    num_features = data.shape[1]
    print(data.shape, num_rows, num_features)

    print("computing mean")
    mean = np.mean(data, axis=0)
    print(mean.shape)
    for i in range(num_features):
        data[i, :] -= mean[i]

    print("normalizing data")
    std_dev = np.zeros(num_features)
    for f in range(num_features):
        #std_dev[f] = np.sqrt(np.sum(np.square(data[:, f]))/num_rows)
        std_dev[f] = np.sum(np.square(data[:, f]))

    # data /= std_dev[:, None] <- to large
    #for i in range(num_rows):
    #    data[i, :] /= std_dev

    print("computing coeff")
    res = np.zeros((num_features, num_features))
    for f1 in range(num_features):
        if f1 % 10 == 0:
            print(f"{f1}/{num_features}")
        for f2 in range(f1, num_features):
            corr = np.dot(data[:, f1], data[:, f2]) / np.sqrt((std_dev[f1]*std_dev[f2]))
            res[f1, f2] = corr
            res[f2, f1] = corr
    return res, mean, std_dev

for level in range(num_levels):
    #collect all data for a given level
    print(f"starting level {level}")

    out_corr_reluX, x1X, flowX, lX = extractLevel(load_sample(all_sample_filenamesA[0]), level)
    print(">>",out_corr_reluX.shape, x1X.shape, flowX.shape)
    x_shape = flatten_merge(out_corr_reluX, x1X, flowX).shape
    pixel_per_image = x_shape[1]
    num_datapoints = len(all_sample_filenamesA) * pixel_per_image
    num_features = x_shape[0]
    #a_full = np.zeros((num_datapoints, num_features))
    #b_full = np.zeros((num_datapoints, num_features))
    full = np.zeros((num_datapoints, num_features*2))
    print(f"{full.shape} full data size")

    # build up matrix containing all features from all images at the current level
    for ii in range(len(all_sample_filenamesA)):
        if ii % 10 == 0:
            print(f"loading data point {ii}")
        out_corr_reluA, x1A, flowA, lA = extractLevel(load_sample(all_sample_filenamesA[ii]), level)
        out_corr_reluB, x1B, flowB, lB = extractLevel(load_sample(all_sample_filenamesB[ii]), level)
        #a_full[ii*x_shape[1]:(ii+1)*x_shape[1], :] = flatten_merge(out_corr_reluA, x1A, flowA).T
        #b_full[ii*x_shape[1]:(ii+1)*x_shape[1], :] = flatten_merge(out_corr_reluB, x1B, flowB).T
        full[ii*pixel_per_image:(ii+1)*pixel_per_image, :num_features] = flatten_merge(out_corr_reluA, x1A, flowA).T
        full[ii*pixel_per_image:(ii+1)*pixel_per_image, num_features:] = flatten_merge(out_corr_reluB, x1B, flowB).T

    print("computing correlation")
    corr, mean, std_dev = custom_coeff(full)
    #corr = custom_coeff(resulting, full[0:1000, :])
    #corr = np.corrcoef(a_full, b_full)
    np.save(f"{resulting}/{level}_corr", corr, allow_pickle=False)
    np.save(f"{resulting}/{level}_mean", mean, allow_pickle=False)
    np.save(f"{resulting}/{level}_std_dev", std_dev, allow_pickle=False)
    print(f"level {level} - done")

    del corr, mean, std_dev, full

print("done")
