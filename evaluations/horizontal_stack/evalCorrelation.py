import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


import os
from glob import glob
import numpy as np
from evaluations.horizontal_stack.tools import load_sample

sample_interface_pathA = "/data/vimb01/evaluations/chairs_PWCNet-20191121-171532_onThings_interface/*"
sample_interface_pathB = "/data/vimb01/evaluations/things_PWCNet-20191122-152857_incomplete_onThings_interface/*"

resulting = "/data/vimb01/evaluations/corr_chairs_thingsInc"

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

SPLITROWS = 200
def custom_coeff(resulting, data):
    num_rows = data.shape[0]

    print("computing mean")
    mean = np.mean(data, axis=1)
    for i in range(num_rows):
        data[i, :] -= mean[i]

    print("normalizing data")
    std_res = np.memmap(resulting+"_std_var_temp", "float64", mode="w+", shape=data.shape)
    std_res[:, :] = data
    std_res *= std_res
    std_dev = np.sqrt(np.sum(std_res, axis=1))

    # data /= std_dev[:, None] <- to large
    for i in range(num_rows):
        data[i, :] /= std_dev[i]

    print("computing coeff")
    res = np.memmap(resulting, "float64", mode="w+", shape=data.shape)
    for r in range(0, num_rows, SPLITROWS):
        print(f"row {r}/{num_rows}")
        for c in range(0, num_rows, SPLITROWS):
            r1 = r + SPLITROWS
            c1 = c + SPLITROWS
            chunk1 = data[r:r1]
            chunk2 = data[c:c1]
            res[r:r1, c:c1] = np.dot(chunk1, chunk2.T)
    return res

for level in range(1): #num_levels):
    level=4
    #collect all data for a given level

    out_corr_reluX, x1X, flowX, lX = extractLevel(load_sample(all_sample_filenamesA[0]), level)
    x_shape = flatten_merge(out_corr_reluX, x1X, flowX).shape
    pixel_per_image = x_shape[1]
    num_datapoints = len(all_sample_filenamesA) * pixel_per_image
    num_features = x_shape[0]
    #a_full = np.zeros((num_datapoints, num_features))
    #b_full = np.zeros((num_datapoints, num_features))
    full = np.zeros((num_datapoints, x_shape[0]*2))
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
    res = custom_coeff(resulting, full.T)
    print(res.shape)
    #corr = np.corrcoef(a_full, b_full)
    #np.save(resulting, corr, allow_pickle=False)

    print(f"level {level} - done")
    exit()

print("done")
