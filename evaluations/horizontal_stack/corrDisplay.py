import matplotlib.pyplot as plt
import numpy as np
import imageio

num_levels = 5

fig = plt.figure()

corr_dir = "/data/vimb01/evaluations/corr_Athings_Cthings"

for i in range(num_levels):
    print(f"level {i}")
    corr_file = f"{corr_dir}/{i}_corr.npy"
    mean_file = f"{corr_dir}/{i}_mean.npy"
    std_dev_file = f"{corr_dir}/{i}_std_dev.npy"

    corr = np.load(corr_file)
    mean = np.load(mean_file)
    std_dev = np.load(std_dev_file)

    print("has NaN: ", np.isnan(corr).any())
    print("min/max (absolute) correlation: ", np.min(np.abs(corr)), np.max(corr))

    print("means:", mean)
    print("std_dev:", std_dev)

    ax = fig.add_subplot(1, num_levels, i+1)
    ax.title.set_text(f"interface {i}")
    plt.imshow(np.abs(corr), cmap="gray")

    imageio.imwrite(f"{corr_dir}/interface_{i}.png", np.clip(np.abs(corr), 0.0, 1.0))

plt.tight_layout()
plt.show()
