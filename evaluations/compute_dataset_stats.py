do_plot = True

import os
import psutil

# 8 processors -> 4 workers with 2 threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

phys_cpus = psutil.cpu_count(logical=False)
#num_procs = int(phys_cpus)  # set 1 worker for every cpu -> reduce OMP threads to 1!
num_procs = 4

import math
import numpy as np

if do_plot:
    import matplotlib.pyplot as plt
from flowbias.utils.localstorage import LocalStorage
from flowbias.datasets import FlyingChairsTrain, FlyingChairsFull, FlyingThings3dCleanTrain, KittiComb2015Train, SintelTrainingCleanTrain
from multiprocessing import Pool
from flowbias.config import Config

pi = np.pi
twopi = 2 * np.pi

chairs_root = "/data/dataB/datasets/FlyingChairs_release/data/"
things_root = "/data/dataB/datasets/FlyingThings3D_subset/"
kitti_root = "/data/dataB/datasets/KITTI_data_scene_flow/"
sintel_root = "/data/dataB/datasets/MPI-Sintel-complete/"

datasets = {
    "flyingChairsTrain": [FlyingChairsTrain, "/data/dataB/datasets/FlyingChairs_release/data/"],
    "flyingChairsFull": [FlyingChairsFull, "/data/dataB/datasets/FlyingChairs_release/data/"],
    "flyingThingsTrain": [FlyingThings3dCleanTrain, "/data/dataB/datasets/FlyingThings3D_subset/"],
    "kittiTrain": [KittiComb2015Train, "/data/dataB/datasets/KITTI_data_scene_flow/", {"preprocessing_crop": False}],
    "sintelTrain": [SintelTrainingCleanTrain, "/data/dataB/datasets/MPI-Sintel-complete/"]
}
sz = 1500
dataset_name = "kitti"  # "flyingChairsFull"

def markXAxis(a):
    a[100: 200, 10: 20] = 1.0  # mark x axis


def pct(a, percentile):
    sa = np.sort(a[a != 0].flatten())
    return sa[int(len(sa) * percentile / 100)]


def log_index_reverse(y):
    return ((y/100)-10)**10


def log_index_fwd(x):
    return (10+np.log10(x))*100


def compute_matrices(id_range):
    id_a = id_range[0]
    id_b = id_range[1]

    dataset_data = datasets[dataset_name]
    args = dataset_data[2] if len(dataset_data) > 2 else {}
    dataset = dataset_data[0]({}, dataset_data[1], photometric_augmentations=False, **args)

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
    if not LocalStorage.contains("field"+dataset_name):

        dataset_data = datasets[dataset_name]
        args = dataset_data[2] if len(dataset_data) > 2 else {}
        dataset = dataset_data[0]({}, dataset_data[1], photometric_augmentations=False, **args)

        field = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)
        logField = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)  # sub 10 flow in
        rstat = np.zeros(1500, np.int)
        logstat = np.zeros(3000, np.int)
        ahisto = np.zeros(int(2 * np.pi * 100), np.int)

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

        LocalStorage.set("field"+dataset_name, field)
        LocalStorage.set("logfield"+dataset_name, logField)
        LocalStorage.set("rstat"+dataset_name, rstat)
        LocalStorage.set("rlogstat"+dataset_name, logstat)
        LocalStorage.set("ahisto" + dataset_name, ahisto)
    else:
        field = LocalStorage.get("field"+dataset_name)
        logField = LocalStorage.get("logfield"+dataset_name)
        rstat = LocalStorage.get("rstat"+dataset_name)
        logstat = LocalStorage.get("rlogstat" + dataset_name)
        ahisto = LocalStorage.get("ahisto" + dataset_name)

        sz = (field.shape[0] - 1) // 2


    field = field.astype(np.float)
    logField = logField.astype(np.float)

    field /= pct(field, 90)
    #logField /= np.percentile(logField, 90)
    logField /= pct(logField, 90)
    #field = np.array(field, dtype=float) / np.mean(field)
    #field = np.clip(np.array(field, dtype=float) / 100, 0.0, 1.0)
    #field = np.array(field, dtype=float) / 100
    #print(">>", field.dtype, np.min(field), np.max(field))
    #print(np.max(field), np.unravel_index(np.argmax(field), field.shape), np.mean(field))
    #fieldd = np.copy(field)
    #fieldd[sz-4:sz+5, sz-4:sz+5] = 0
    #field /= np.max(fieldd)
    #print(np.max(field), np.unravel_index(np.argmax(field), field.shape))

    if do_plot:
        print(field.shape)
        window = 500
        field = field[sz-window:sz+window, sz-window:sz+window]
        markXAxis(field)
        print(field.shape)
        plt.figure()
        plt.title(dataset_name)
        plt.imshow(field.T, cmap="hot", vmin=0, vmax=1)
        img_rng = [0-10, window, 2*window+10]
        ltrns = [str(tick) for tick in [-window, 0, window]]
        plt.xticks(img_rng, ltrns)
        plt.yticks(img_rng, ltrns)
        plt.ylim(0, 2 * window)
        plt.xlim(0, 2 * window)
        plt.savefig(Config.temp_directory+f"dataset_stats/{dataset_name}_flow_abs.png", dpi=800, bbox_inches="tight")
        #plt.show()

        p = (10 + np.log10(1)) * 100
        print("LL>", sz, p)
        logField[sz + int(p):sz + int(p)+10, :] += 0.2
        logField[sz - int(p):sz - int(p)+10, :] += 0.2
        logField[:, sz + int(p):sz + int(p)+10] += 0.2
        logField[:, sz - int(p):sz - int(p)+10] += 0.2
        markXAxis(logField)
        #logField[int(xa) + sz, int(ya) + sz] += 1

        plt.figure()
        plt.title(dataset_name+"_log")
        plt.imshow(logField.T, cmap="hot", vmin=0, vmax=1)
        pts = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=np.double)
        pts_mirr = np.concatenate([-pts[::-1], [1234], pts])
        labels = ['%.0E' % abs(v) for v in pts_mirr]
        img_rng = np.array([log_index_fwd(abs(i))*np.sign(i) for i in pts_mirr])
        img_rng[len(img_rng) // 2] = 0
        labels[len(labels) // 2] = "0.0"
        print(img_rng)
        plt.xticks(img_rng + sz, labels)
        plt.yticks(img_rng + sz, labels)
        plt.ylim(0, 2*sz)
        plt.xlim(0, 2*sz)
        plt.savefig(Config.temp_directory + f"dataset_stats/{dataset_name}_flow_log.png", dpi=800, bbox_inches="tight")
        #plt.show()

        plt.figure()
        plt.title(dataset_name+" rstat")
        plt.plot(range(len(rstat)), rstat)
        #plt.xlim(0, len(rstat))
        plt.xlim(-10, 100)
        #plt.ylim(0, 500)
        plt.savefig(Config.temp_directory + f"dataset_stats/{dataset_name}_rstat.png", bbox_inches="tight")
        #plt.show()
        print("mode", np.argmax(rstat), rstat[np.argmax(rstat)], rstat[600])
        print("!!", rstat[0], logstat[0])

        # normalize
        logstat = logstat.astype(np.double) / np.sum(logstat)

        plt.figure()
        plt.title(dataset_name+" r log stat")
        print("log mode", ((np.argmax(logstat)/100)-10)**10)
        plt.plot(range(len(logstat)), logstat)
        #plt.xlim(0, len(logstat))
        #ldisp = [950, 1200]
        #plt.xlim(ldisp[0], ldisp[1])
        #plt.ylim(0, 0.0065)
        #plt.ylim(-1e6, 3e7)
        #xr = range(ldisp[0], ldisp[1]+1, 25)
        #yr = [f"{log_index_reverse(l):.3f}" for l in xr]
        #plt.xticks(xr, yr)

        pts = np.array([1234, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0], dtype=np.double)
        labels = ['%.0E' % abs(v) for v in pts]
        img_rng = np.array([log_index_fwd(abs(i)) * np.sign(i) for i in pts])
        print("LLR>", img_rng)
        img_rng[0] = 0
        labels[0] = "0.0"
        plt.xticks(img_rng, labels)
        plt.xlim(0, img_rng[-1])

        plt.savefig(Config.temp_directory + f"dataset_stats/{dataset_name}_log_stat.png", dpi=400, bbox_inches="tight")
        #plt.show()


        cs = np.cumsum(np.array(rstat, dtype=np.float))
        cs /= cs[-1]

        plt.figure()
        plt.title(dataset_name+" cumsum")
        plt.plot(range(len(cs)), cs)
        plt.xlim(0, 250)
        #plt.ylim(0.8, 1)
        plt.savefig(Config.temp_directory + f"dataset_stats/{dataset_name}_cumsum.png", bbox_inches="tight")
        #plt.show()

        # show as intensity ring?
        plt.figure()
        plt.title(dataset_name + "ahisto")
        plt.plot(range(len(ahisto)), ahisto)
        plt.xlim(0, len(rstat))
        plt.savefig(Config.temp_directory + f"{dataset_name}_ahisto.png")
        # plt.show()


    print("done")
