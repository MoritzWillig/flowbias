import matplotlib.pyplot as plt
import numpy as np
import imageio

num_levels = 5

fig = plt.figure()

corr_dir = "/visinf/home/vimb01/projects/evals/results_A_B/"

for i in range(num_levels):
    print(f"level {i}")
    corr_file = f"{corr_dir}/{i}_corr.npy"

    corr = np.load(corr_file)

    print("has NaN: ", np.isnan(corr).any())
    print("min/max (absolute) correlation: ", np.min(np.abs(corr)), np.max(corr))

    ax = fig.add_subplot(1, num_levels, i+1)
    ax.title.set_text(f"interface {i}")
    plt.imshow(np.abs(corr), cmap="gray")

    #imageio.imwrite(f"{corr_dir}/interface_{i}.png", np.abs(corr))

plt.tight_layout()
plt.show()
