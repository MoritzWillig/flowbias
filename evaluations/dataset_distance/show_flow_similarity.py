from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter

from flowbias.utils.localstorage import LocalStorage


def preprocess_field(field, log_field):
    #z = np.log(field+1e-10)
    p = np.log(log_field+1)
    p = median_filter(p, size=5)
    p = gaussian_filter(p, sigma=2)

    print("sum", np.sum(p))
    p /= np.sum(p)

    return p

def visualize(dataset_name_A, dataset_name_B):
    field = LocalStorage.get("field_"+dataset_name_A)
    log_field = LocalStorage.get("logfield_"+dataset_name_A)

    z = preprocess_field(field, log_field)
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

visualize("flyingChairsTrain", None)