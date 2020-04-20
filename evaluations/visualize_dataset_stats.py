do_plot = True

import numpy as np
import matplotlib.pyplot as plt
from flowbias.utils.localstorage import LocalStorage
from flowbias.utils.meta_infrastructure import get_dataset_names
from flowbias.datasets import FlyingChairsTrain, FlyingChairsFull, FlyingThings3dCleanTrain, KittiComb2015Train, SintelTrainingCleanTrain
from flowbias.config import Config

pi = np.pi
twopi = 2 * np.pi


def mark_x_axis(a):
    """
    puts a bar in the direction of the x axis, in the lower left corner of the image
    :param a:
    :return:
    """
    a[100:200, 10:20] = 1.0  # mark x axis


def pct(a, percentile):
    sa = np.sort(a[a != 0].flatten())
    return sa[int(len(sa) * percentile / 100)]


def log_index_reverse(y):
    return ((y/100)-10)**10


def log_index_fwd(x):
    return (10+np.log10(x))*100


dataset_stats = {}


def load_dataset_stats(dataset_name):
    if not LocalStorage.contains("field_"+dataset_name):
        print(f"no data for {dataset_name}")
    else:
        print(f"found {dataset_name}")
        field = LocalStorage.get("field_"+dataset_name)
        logField = LocalStorage.get("logfield_"+dataset_name)
        rstat = LocalStorage.get("rstat_"+dataset_name)
        logstat = LocalStorage.get("rlogstat_" + dataset_name)
        ahisto = LocalStorage.get("ahisto_" + dataset_name)
        dataset_stats[dataset_name] = (field, logField, rstat, logstat, ahisto)


def compute_for_dataset(dataset_name, prefix):
    if dataset_name not in dataset_stats:
        print(f"skipping {dataset_name}")
        return
    print(f"visualizing {dataset_name}")
    field, logField, rstat, logstat, ahisto = dataset_stats[dataset_name]

    sz = (field.shape[0] - 1) // 2

    #field[sz-4:sz+5, sz-4:sz+5] = 0
    field = field.astype(np.float)
    logField = logField.astype(np.float)

    field /= pct(field, 90)
    logField /= pct(logField, 90)
    #print("$$>>", field.max(), field.sum())
    #field /= field.sum()
    #logField /= logField.sum()
    #print("$$>>", field.max(), field.sum())
    #logField /= np.percentile(logField, 90)
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
        mark_x_axis(field)
        #field /= field.max()
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
        plt.savefig(Config.temp_directory+f"dataset_stats/{prefix}{dataset_name}_flow_abs.png", dpi=800, bbox_inches="tight")
        #plt.show()

        p = (10 + np.log10(1)) * 100
        print("LL>", sz, p)
        logField[sz + int(p):sz + int(p)+10, :] += 0.2
        logField[sz - int(p):sz - int(p)+10, :] += 0.2
        logField[:, sz + int(p):sz + int(p)+10] += 0.2
        logField[:, sz - int(p):sz - int(p)+10] += 0.2
        mark_x_axis(logField)
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
        plt.savefig(Config.temp_directory + f"dataset_stats/{prefix}{dataset_name}_flow_log.png", dpi=800, bbox_inches="tight")
        #plt.show()

        plt.figure()
        plt.title(dataset_name+" rstat")
        plt.plot(range(len(rstat)), rstat)
        #plt.xlim(0, len(rstat))
        plt.xlim(-10, 100)
        #plt.ylim(0, 500)
        plt.savefig(Config.temp_directory + f"dataset_stats/{prefix}{dataset_name}_rstat.png", bbox_inches="tight")
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

        plt.savefig(Config.temp_directory + f"dataset_stats/{prefix}{dataset_name}_log_stat.png", dpi=400, bbox_inches="tight")
        #plt.show()


        cs = np.cumsum(np.array(rstat, dtype=np.float))
        cs /= cs[-1]

        plt.figure()
        plt.title(dataset_name+" cumsum")
        plt.plot(range(len(cs)), cs)
        plt.xlim(0, 250)
        #plt.ylim(0.8, 1)
        plt.savefig(Config.temp_directory + f"dataset_stats/{prefix}{dataset_name}_cumsum.png", bbox_inches="tight")
        #plt.show()

        ahisto = ahisto.astype(np.float) / np.sum(ahisto)
        plt.figure()
        plt.title(dataset_name + "ahisto")
        plt.plot(range(len(ahisto)), ahisto)
        plt.xlim(0, 628)
        plt.xticks([0, 157, 314, 471, 628], [r"$-\pi$", r"$-\pi/2$", 0, r"$\pi/2$", r"$\pi$"])
        plt.savefig(Config.temp_directory + f"dataset_stats/{prefix}{dataset_name}_ahisto.png")
        # plt.show()

        plt.close('all')


if not do_plot:
    print("plotting disabled")

#dataset_names = get_dataset_names()
#prefix = ""
dataset_names = [
    "H_flyingChairsValid", "H_flyingThingsCleanValid", "H_kitti2015Valid", "H_sintelCleanValid", "H_sintelFinalValid",
    "I_flyingChairsValid", "I_flyingThingsCleanValid", "I_kitti2015Valid", "I_sintelCleanValid", "I_sintelFinalValid",
    "pwc_chairs_flyingChairsValid", "pwc_chairs_flyingThingsCleanValid", "pwc_chairs_kitti2015Valid", "pwc_chairs_sintelCleanValid", "pwc_chairs_sintelFinalValid",
    "pwc_kitti_flyingChairsValid", "pwc_kitti_flyingThingsCleanValid", "pwc_kitti_kitti2015Valid", "pwc_kitti_sintelCleanValid", "pwc_kitti_sintelFinalValid",
    "pwcWOX1_chairs_flyingChairsValid", "pwcWOX1_chairs_flyingThingsCleanValid", "pwcWOX1_chairs_kitti2015Valid", "pwcWOX1_chairs_sintelCleanValid", "pwcWOX1_chairs_sintelFinalValid",
    "pwcWOX1_kitti_flyingChairsValid", "pwcWOX1_kitti_flyingThingsCleanValid", "pwcWOX1_kitti_kitti2015Valid", "pwcWOX1_kitti_sintelCleanValid", "pwcWOX1_kitti_sintelFinalValid",
    "pwcWOX1_on_CTSK_flyingChairsValid", "pwcWOX1_on_CTSK_flyingThingsCleanValid", "pwcWOX1_on_CTSK_kitti2015Valid", "pwcWOX1_on_CTSK_sintelCleanValid", "pwcWOX1_on_CTSK_sintelFinalValid",
    "pwcWOX1_sintel_flyingChairsValid", "pwcWOX1_sintel_flyingThingsCleanValid", "pwcWOX1_sintel_kitti2015Valid", "pwcWOX1_sintel_sintelCleanValid", "pwcWOX1_sintel_sintelFinalValid",
    "pwcWOX1_things_flyingChairsValid", "pwcWOX1_things_flyingThingsCleanValid", "pwcWOX1_things_kitti2015Valid", "pwcWOX1_things_sintelCleanValid", "pwcWOX1_things_sintelFinalValid"
]


for dataset_name in dataset_names:
    load_dataset_stats(dataset_name)

for dataset_name in dataset_names:
    prefix = "models/"
    compute_for_dataset(dataset_name, prefix)

print("done")
