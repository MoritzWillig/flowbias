import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import zoom

from flowbias.utils.data_loading import load_sample
from flowbias.utils.visualization.AGrid import AGrid

font = ImageFont.truetype("/home/moritz/.local/share/fonts/RobotoMono-Bold.ttf", 40, encoding="unic")

models = [
    "/data/dataA/model_interfaces/wox1/pwcWOX1_chairs_flyingChairsSubset/",
    "/data/dataA/model_interfaces/wox1/pwcWOX1_things_flyingChairsSubset/",
    "/data/dataA/model_interfaces/wox1/pwcWOX1_sintel_flyingChairsSubset/"
]

extend = 0
span = 2*extend + 1

out_corr_relu = []
x1 = []
x2 = []
x2_warp = []
flow = []
for model in models:
    out_corr_relu_i, x1_i, x2_i, x2_warp_i, flow_i, l = load_sample(model+"0.npz")
    out_corr_relu.append(out_corr_relu_i)

num_layers = len(out_corr_relu[0])
print(">>", num_layers)

grid = AGrid(
    (span*len(models), span*num_layers),
    (out_corr_relu[0][-1].shape[2] + 200, out_corr_relu[0][-1].shape[3] + 200),
    text_params={"font": font},
    title_params={"font": font}, text_height=0)

for m, c in enumerate(out_corr_relu):
    for l in range(num_layers):
        for i in range(81):
            x = i % 9
            y = i // 9

            if (not (4-extend<=x<=4+extend)) or (not (4-extend<=y<=4+extend)):
                continue

            ci = c[-(l+1)][0, i, :, :]
            print(l, ci.shape)
            ci = zoom(ci, 2**(l+1), order=0)
            #ci /= np.linalg.norm(ci, ord=2)
            print("!>", m,l, np.mean(ci), ci.shape)
            ci -= np.mean(ci)
            ci /= np.std(ci)
            ci += np.mean(ci)
            ci = np.clip(ci, 0, 1)
            #print(np.max(ci))

            grid.place((m*span) + x-4+extend, (l*span) + y-4+extend, ci)

plt.imshow(grid.get_image())
plt.show()