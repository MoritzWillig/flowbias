import os
import psutil
import sys
import time

# using threads on a ryzen 1900x is faster by a factor of 3
use_threading = True
force_recompute = True

if use_threading:
    # 8 processors -> 4 workers with 2 threads
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    phys_cpus = psutil.cpu_count(logical=False)
    #num_procs = int(phys_cpus)  # set 1 worker for every cpu -> reduce OMP threads to 1!
    num_procs = 4

import math
import numpy as np
import numpy.ma as ma

from flowbias.datasets import FlowOnlyNpDataset
from flowbias.evaluations.log_transforms import log_index_fwd, log_index_reverse

from flowbias.utils.meta_infrastructure import get_available_datasets
from flowbias.utils.localstorage import LocalStorage
from multiprocessing import Pool

pi = np.pi
twopi = 2 * np.pi

assert (len(sys.argv) == 2)
dataset_name = sys.argv[1]
#dataset_name = "kitti2015Valid"  # "flyingChairsFull"
#dataset_name = "@/data/dataB/temp/predictedFlows/pwcWOX1_on_CTSK_flyingChairsValid"

if dataset_name[0] != "@":
    datasets = get_available_datasets(force_mode="test", restrict_to=[dataset_name])
else:
    flow_dataset = FlowOnlyNpDataset({}, dataset_name[1:])
    dataset_name = os.path.basename(dataset_name[1:])
    datasets = {
        dataset_name: flow_dataset
    }

assert(len(datasets) == 1)

field_extend = 1500
field_size = (2 * field_extend) + 1

rstat_bins = 1500
logstat_bins = 3000
ahisto_bins = int(twopi * 100)


def compute_matrices(id_range):
    id_a = id_range[0]
    id_b = id_range[1]

    dataset = datasets[dataset_name]

    field = np.zeros((field_size, field_size), np.int)
    log_field = np.zeros((field_size, field_size), np.int)
    rstat = np.zeros(rstat_bins, np.int)
    logstat = np.zeros(logstat_bins, np.int)
    ahisto = np.zeros(ahisto_bins, np.int)  # angle histogram, hundreds of degree


    for i in range(id_a, id_b):
        sample = dataset[i]

        flow = np.transpose(sample["target1"].cpu().detach().numpy(), (1, 2, 0))
        if "input_valid" in sample:
            mask = sample["input_valid"].cpu().detach().numpy().astype(np.bool).squeeze()
            flow = flow[mask]
        else:
            flow = flow.reshape(-1, 2)

        xx = flow[:, 0]
        yy = flow[:, 1]
        r = np.sqrt(xx ** 2 + yy ** 2)  # radius
        a = np.arctan2(yy, xx)  # angle [-pi, +pi]

        has_flow_selector = r > 1e-10
        num_zero_flow = r.size - np.count_nonzero(has_flow_selector)

        # write absolute stats (rstat)
        rstat_part, _ = np.histogram(r, rstat_bins, (0, rstat_bins))
        rstat += rstat_part

        # write angle histogram
        an = ((a[has_flow_selector] + twopi) * 100).astype(np.int) % int(100 * twopi)  # to range [0, 2PI] * 100
        ahisto_part, _ = np.histogram(an, ahisto_bins, (0, ahisto_bins))
        ahisto += ahisto_part

        # log_stat histogram
        log_r = log_index_fwd(r[has_flow_selector])
        log_stat_part, _ = np.histogram(log_r, logstat_bins, (0, logstat_bins))
        logstat += log_stat_part
        logstat[0] += num_zero_flow

        # absolute flow vector histogram
        field_part, _, _ = np.histogram2d(
            xx, yy,
            [field_size, field_size],
            [[-field_extend, field_extend], [-field_extend, field_extend]])
        field += field_part.astype(np.int)

        # log flow vector histogram
        selected_a = a[has_flow_selector]
        log_x = np.cos(selected_a) * log_r
        log_y = np.sin(selected_a) * log_r
        log_field_part, _, _ = np.histogram2d(
            log_x, log_y,
            [field_size, field_size],
            [[-field_extend, field_extend], [-field_extend, field_extend]])
        log_field += log_field_part.astype(np.int)
        log_field[field_extend, field_extend] += num_zero_flow

    return field, log_field, rstat, logstat, ahisto


if __name__ == '__main__':
    print(f"computing dataset stats: {dataset_name}")
    if (not LocalStorage.contains("field_"+dataset_name)) or force_recompute:
        dataset = datasets[dataset_name]

        start = time.time()

        print(f"using threads: {use_threading}")
        if use_threading:
            field = np.zeros((field_size, field_size), np.int)  # count of flow vectors
            logField = np.zeros((field_size, field_size), np.int)  # count of log10 flow vectors
            rstat = np.zeros(rstat_bins, np.int)  # statistic of flow vector magnitudes
            logstat = np.zeros(logstat_bins, np.int)  # statistic of log10 flow vector magnitudes
            ahisto = np.zeros(ahisto_bins, np.int)  # flow direction

            with Pool(processes=num_procs) as p:
                print(f"{len(dataset)} samples using {num_procs} processes ({phys_cpus} processors)")
                proc_range = int(math.ceil(len(dataset) / num_procs))

                chunks = [[proc_id*proc_range, (proc_id+1)*proc_range] for proc_id in range(num_procs)]
                chunks[-1][1] = len(dataset)

                results = p.map(compute_matrices, chunks)

            #print(len(results), len(results[0]))
            for i in range(num_procs):
                field += results[i][0]
                logField += results[i][1]
                rstat += results[i][2]
                logstat += results[i][3]
                ahisto += results[i][4]
        else:
            field, logField, rstat, logstat, ahisto = compute_matrices([0, len(dataset)])

        end = time.time()
        print(f"computing dataset stats for {dataset_name} with {len(dataset)} took {end - start}s")

        LocalStorage.set("field_"+dataset_name, field)
        LocalStorage.set("logfield_"+dataset_name, logField)
        LocalStorage.set("rstat_"+dataset_name, rstat)
        LocalStorage.set("rlogstat_"+dataset_name, logstat)
        LocalStorage.set("ahisto_" + dataset_name, ahisto)
    else:
        print(f"existing data for {dataset_name}")

    print("done")
