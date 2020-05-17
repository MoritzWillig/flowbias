from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.stats import entropy
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score

from flowbias.utils.localstorage import LocalStorage
from flowbias.utils.meta_infrastructure import get_dataset_names
from flowbias.evaluations.log_transforms import log_index_fwd, log_index_reverse

field_type = "abs"


def inc_mi(x, y):
    xy = (x+y).astype(np.double) + 1e-10
    xy = xy / np.sum(xy)

    x = x.astype(np.double) + 1e-10
    y = y.astype(np.double) + 1e-10
    x = x / np.sum(x)
    y = y / np.sum(y)

    print(np.min(xy), np.min(x), np.min(y))

    mi_subs = []
    for i in range(x.shape[0]):
        mi_sub = np.sum(xy[i,:] * np.log(xy[i,:]/(x[i, :] * y[i, :])))
        mi_subs.append(mi_sub)
    mi = np.sum(mi_subs)
    return mi


def load_and_preprocess_field(dataset_name):
    field = LocalStorage.get("field_" + dataset_name)
    s = (field.shape[0] // 2)
    window = 400
    field = field[s - window:s + window, s - window:s + window]

    field = median_filter(field, size=5)
    field = gaussian_filter(field, sigma=2)

    #field = np.round(np.log10(field + 1)*100)+1e-10 # use for adjusted mutal information
    return field


def load_and_preprocess_log_field(dataset_name):
    log_field = LocalStorage.get("logfield_" + dataset_name)
    #z = np.log(field+1e-10)

    #field = log_field[1200:1801, 1200:1801]

    p = log_field.astype(np.double)
    #p = np.log(log_field+1)
    p = median_filter(p, size=5)
    p = gaussian_filter(p, sigma=2)

    # only take the 'ring' of radius [1e-04, 1e+03]
    s = (log_field.shape[0] // 2)
    indices = np.indices(log_field.shape, dtype=np.float) - s
    r_map = np.sqrt((indices[0, :, :]**2) + (indices[1, :, :]**2))

    rev_inner = log_index_fwd(1e-04)
    rev_outer = log_index_fwd(1e+03)

    sel_p = p[np.logical_and(r_map >= rev_inner, r_map <= rev_outer)]
    p = sel_p
    #print(f"{dataset_name} sum", np.sum(p), p.shape, np.sum(sel_p), sel_p.shape)
    p /= np.sum(p)
    return p


def visualize(dataset_name_A, dataset_name_B):
    z_a = load_and_preprocess_field(dataset_name_A)
    z_b = load_and_preprocess_field(dataset_name_B)

    z_a += 1e-10
    z_b += 1e-10

    kl_ab = entropy(z_a.ravel(), z_b.ravel())
    kl_ba = entropy(z_b.ravel(), z_a.ravel())
    print(f"kl[{dataset_name_A},{dataset_name_B}]: {kl_ab} {kl_ba}")

    z = z_a
    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    print(np.max(z))

    # show hight map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap="hot", vmin=0, vmax=5e-7, rcount=100)
    ax.set_xlim3d(1200, 1800)
    ax.set_ylim3d(1200, 1800)
    ax.set_zlim3d(0, 1e-6)
    plt.title('z as 3d height map')
    plt.show()

#visualize("flyingChairsTrain", "flyingThingsCleanTrain")

# flows of sintel clean and sintel final are the same, so we exclude one
datasets = get_dataset_names(select_by_any_tag=["train"], exclude_by_tag=["clean"])
print("datasets:", datasets)

if field_type == "log":
    bins = np.logspace(-10, -3, num=100, base=10, endpoint=True)

kl = np.zeros((len(datasets), len(datasets)))
for i, dataset_a in enumerate(datasets):
    if field_type == "log":
        p_a = load_and_preprocess_log_field(dataset_a)
        d_a = np.digitize(p_a, bins)
    else:
        p_a = load_and_preprocess_field(dataset_a)
        #d_a = p_a.ravel()
        d_a = p_a
    #p_a += 1e-10


    for j, dataset_b in enumerate(datasets):
        if field_type == "log":
            p_b = load_and_preprocess_log_field(dataset_b)
            d_b = np.digitize(p_b, bins)
        else:
            p_b = load_and_preprocess_field(dataset_b)
            #d_b = p_b.ravel()
            d_b = p_b
        print(np.max(p_a), np.max(p_b), d_a.shape, d_b.shape)
        #kl[i, j] = adjusted_mutual_info_score(d_a, d_b, average_method="arithmetic")
        kl[i, j] = inc_mi(d_a, d_b)
        #print(np.min(d_a), np.min(d_b), np.sum(d_a), np.sum(d_b))
        #kl[i, j] = entropy((d_a/np.sum(d_a)).astype(np.double), (d_b/np.sum(d_b)).astype(np.double))
        print(kl[i, j])

print(kl)
