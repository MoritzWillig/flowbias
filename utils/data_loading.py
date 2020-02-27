import numpy as np


def load_sample(filename):
    npzfile = np.load(filename)
    num_levels = len(npzfile.files) // 6

    out_corr_relu = [npzfile["out_corr_relu_"+str(i)] for i in range(num_levels)]
    x1 = [npzfile["x1_"+str(i)] for i in range(num_levels)]
    x2 = [npzfile["x2_"+str(i)] for i in range(num_levels)]
    x2_warp = [npzfile["x2_warp_"+str(i)] for i in range(num_levels)]
    flow = [npzfile["flow_"+str(i)] for i in range(num_levels)]
    l = [npzfile["l_"+str(i)] for i in range(num_levels)]

    return out_corr_relu, x1, x2, x2_warp, flow, l


def load_sample_level(filename, level):
    npzfile = np.load(filename)
    level_str = str(level)

    out_corr_relu = npzfile["out_corr_relu_"+level_str]
    x1 = npzfile["x1_"+level_str]
    x2 = npzfile["x2_"+level_str]
    x2_warp = npzfile["x2_warp_"+level_str]
    flow = npzfile["flow_"+level_str]
    l = npzfile["l_"+level_str]

    return out_corr_relu, x1, x2, x2_warp, flow, l
