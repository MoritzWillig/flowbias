import os
import psutil

# 8 processors -> 4 workers with 2 threads
from flowbias.utils.meta_infrastructure import get_available_datasets

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

phys_cpus = psutil.cpu_count(logical=False)
#num_procs = int(phys_cpus)  # set 1 worker for every cpu -> reduce OMP threads to 1!
num_procs = 4

import math
import numpy as np

from flowbias.utils.localstorage import LocalStorage
from multiprocessing import Pool

pi = np.pi
twopi = 2 * np.pi

dataset_name = "kitti2015Valid"  # "flyingChairsFull"
datasets = get_available_datasets(force_mode="test", restrict_to=[dataset_name])
assert(len(datasets) == 1)

sz = 1500


def log_index_reverse(y):
    return ((y/100)-10)**10


def log_index_fwd(x):
    return (10+np.log10(x))*100


def compute_matrices(id_range):
    id_a = id_range[0]
    id_b = id_range[1]

    dataset = datasets[dataset_name]

    field = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)
    logField = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)  # sub 10 flow in
    rstat = np.zeros(1500, np.int)
    logstat = np.zeros(3000, np.int)
    ahisto = np.zeros(int(2*np.pi*100), np.int)  # angle histogram, hundreds of degree

    #s1 = np.zeros(1, dtype=double)

    for i in range(id_a, id_b):
        sample = dataset[i]

        flow = np.transpose(sample["target1"].cpu().detach().numpy(), (1, 2, 0))
        if "input_valid" in sample:
            mask = sample["input_valid"].cpu().detach().numpy().astype(np.bool)
            flow = flow[mask]
        else:
            mask = None

        xx = flow[:, :, 0]
        yy = flow[:, :, 1]
        r = np.sqrt(xx ** 2 + yy ** 2)  # radius
        a = np.arctan2(yy, xx)  # angle [-pi, +pi]

        # print("!!", np.min(r[r!=0.0]))
        for xi, yi, rr, aa in np.nditer((xx, yy, r, a)):
            # write absolute stats
            rstat[int(rr)] += 1

            # write log stats
            if rr < 1e-10:
                logstat[0] += 1
            else:
                logstat[int((10 + np.log10(rr)) * 100)] += 1

            ta = (twopi + aa) % twopi  # to range [0, 2PI]
            ahisto[int(ta*100)] += 1

            # write to log field
            if rr < 1e-10:
                xa = 0
                ya = 0
            else:
                rr = (10 + np.log10(rr)) * 100
                xa = np.cos(aa) * rr
                ya = np.sin(aa) * rr
            logField[sz + int(xa), sz + int(ya)] += 1

            # write to absolute field
            if -sz > xi or xi > sz or -sz > yi or yi > sz:
                continue
            field[sz + int(xi), sz + int(yi)] += 1

    return field, logField, rstat, logstat, ahisto


if __name__ == '__main__':
    if not LocalStorage.contains("field_"+dataset_name):

        dataset = datasets[dataset_name]

        field = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)  # count of flow vectors
        logField = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)  # count of log10 flow vectors
        rstat = np.zeros(1500, np.int)  # statistic of flow vector magnitudes
        logstat = np.zeros(3000, np.int)  # statistic of log10 flow vector magnitudes
        ahisto = np.zeros(int(2 * np.pi * 100), np.int)  # flow direction

        with Pool(processes=num_procs) as p:
            print(f"{len(dataset)} samples using {num_procs} processes ({phys_cpus} processors)")
            proc_range = int(math.ceil(len(dataset) / num_procs))

            chunks = [[proc_id*proc_range, (proc_id+1)*proc_range] for proc_id in range(num_procs)]
            chunks[-1][1] = len(dataset)

            results = p.map(compute_matrices, chunks)

        print(len(results), len(results[0]))
        for i in range(num_procs):
            field += results[i][0]
            logField += results[i][1]
            rstat += results[i][2]
            logstat += results[i][3]
            ahisto += results[i][4]

        LocalStorage.set("field_"+dataset_name, field)
        LocalStorage.set("logfield_"+dataset_name, logField)
        LocalStorage.set("rstat_"+dataset_name, rstat)
        LocalStorage.set("rlogstat_"+dataset_name, logstat)
        LocalStorage.set("ahisto_" + dataset_name, ahisto)
    else:
        print(f"existing data for {dataset_name}")

    print("done")
