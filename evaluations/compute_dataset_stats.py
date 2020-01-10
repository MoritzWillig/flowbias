do_plot = True

import math
import numpy as np

if do_plot:
    import matplotlib.pyplot as plt
from flowbias.utils.localstorage import LocalStorage
from flowbias.datasets import FlyingChairsTrain, FlyingThings3dCleanTrain, KittiComb2015Train, SintelTrainingCleanTrain
from multiprocessing import Pool
import psutil

chairs_root = "/data/dataB/datasets/FlyingChairs_release/data/"
things_root = "/data/dataB/datasets/FlyingThings3D_subset/"
kitti_root = "/data/dataB/datasets/KITTI_data_scene_flow/"
sintel_root = "/data/dataB/datasets/MPI-Sintel-complete/"

datasets = {
    "flyingChairsTrain": [FlyingChairsTrain, "/data/dataB/datasets/FlyingChairs_release/data/"],
    "flyingThingsTrain": [FlyingThings3dCleanTrain, "/data/dataB/datasets/FlyingThings3D_subset/"],
    "kittiTrain": [KittiComb2015Train, "/data/dataB/datasets/KITTI_data_scene_flow/", {"preprocessing_crop": True}],
    "sintelTrain": [SintelTrainingCleanTrain, "/data/dataB/datasets/MPI-Sintel-complete/"]
}
sz = 1500
dataset_name = "flyingThingsTrain"


def pct(a, percentile):
    sa = np.sort(a[a != 0].flatten())
    return sa[int(len(sa) * percentile / 100)]


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

    for i in range(id_a, id_b):
        sample = dataset[i]

        flow = np.transpose(sample["target1"].cpu().detach().numpy(), (1, 2, 0))

        r = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        t = np.arctan2(flow[:, :, 1], flow[:, :, 0])

        # print("!!", np.min(r[r!=0.0]))

        for rr in np.nditer(r):
            # write absolute stats
            rstat[int(rr)] += 1

            # write log stats
            if rr < 1e-10:
                logstat[0] += 1
            else:
                logstat[int((10 + np.log10(rr)) * 100)] += 1

        xx = np.cos(t) * r
        yy = np.sin(t) * r
        for xi, yi, rr, tt in np.nditer((xx, yy, r, t)):
            # write to log field
            if rr < 1e-10:
                xa = 0
                ya = 0
            else:
                rr = (10 + np.log10(rr)) * 100
                xa = np.cos(tt) * rr
                ya = np.sin(tt) * rr
            logField[int(xa) + sz, int(ya) + sz] += 1

            # write to absolute field
            if -sz > xi or xi > sz or -sz > yi or yi > sz:
                continue
            field[int(xi) + sz, int(yi) + sz] += 1
    return field, logField, rstat, logstat


if __name__ == '__main__':
    if not LocalStorage.contains("field"+dataset_name):

        dataset_data = datasets[dataset_name]
        args = dataset_data[2] if len(dataset_data) > 2 else {}
        dataset = dataset_data[0]({}, dataset_data[1], photometric_augmentations=False, **args)

        field = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)
        logField = np.zeros((2 * sz + 1, 2 * sz + 1), np.int)  # sub 10 flow in
        rstat = np.zeros(1500, np.int)
        logstat = np.zeros(3000, np.int)

        cpu_usage = 1
        phys_cpus = psutil.cpu_count(logical=False)
        num_procs = int(phys_cpus * cpu_usage)

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

        LocalStorage.set("field"+dataset_name, field)
        LocalStorage.set("logfield"+dataset_name, logField)
        LocalStorage.set("rstat"+dataset_name, rstat)
        LocalStorage.set("rlogstat"+dataset_name, logstat)
    else:
        field = LocalStorage.get("field"+dataset_name)
        logField = LocalStorage.get("logfield"+dataset_name)
        rstat = LocalStorage.get("rstat"+dataset_name)
        logstat = LocalStorage.get("rlogstat" + dataset_name)

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
        win = 500
        field = field[sz-win:sz+win, sz-win:sz+win]
        print(field.shape)
        plt.figure()
        plt.title(dataset_name)
        plt.imshow(field, cmap="hot", vmin=0, vmax=1)
        img_rng = [0-10, win, 2*win+10]
        ltrns = [str(tick) for tick in [-win, 0, win]]
        plt.xticks(img_rng, ltrns)
        plt.yticks(img_rng, ltrns)
        plt.show()

        #logField[int((10 + np.log10(1e-10)) * 100), :] = 0
        #logField[int((10 + np.log10(-1e-10)) * 100), :] = 0
        plt.figure()
        plt.title(dataset_name+"_log")
        plt.imshow(logField, cmap="hot", vmin=0, vmax=1)
        img_rng = [0-10, sz, 2*sz+10]
        ltrns = [str(tick) for tick in [-sz, 0, sz]]
        plt.xticks(img_rng, ltrns)
        plt.yticks(img_rng, ltrns)
        plt.show()

        plt.figure()
        plt.title(dataset_name+" rstat")
        plt.plot(range(len(rstat)), rstat)
        plt.xlim(0, len(rstat))
        plt.ylim(0, 500)
        plt.show()

        plt.figure()
        plt.title(dataset_name+" r log stat")
        plt.plot(range(len(logstat)), logstat)
        plt.xlim(0, len(logstat))
        plt.ylim(0, 500)
        plt.show()

        cs = np.cumsum(np.array(rstat, dtype=np.float))
        cs /= cs[-1]

        plt.figure()
        plt.title(dataset_name+" cumsum")
        plt.plot(range(len(cs)), cs)
        plt.xlim(0, len(rstat))
        plt.ylim(0.8, 1)
        plt.show()


    print("done")