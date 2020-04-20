import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import zoom
from skimage.draw import line_aa

from flowbias.utils.data_loading import load_sample
from flowbias.utils.visualization.AGrid import AGrid

font = ImageFont.truetype("/home/moritz/.local/share/fonts/RobotoMono-Bold.ttf", 20, encoding="unic")

plot_statistics = False

models = [
    "/data/dataA/model_interfaces/wox1/pwcWOX1_chairs_flyingChairsSubset/",
    "/data/dataA/model_interfaces/wox1/pwcWOX1_things_flyingChairsSubset/",
    "/data/dataA/model_interfaces/wox1/pwcWOX1_sintel_flyingChairsSubset/"
]

#models = [
#    "/data/dataA/model_interfaces/cts_experts/expert_CTS_add01_CC_flyingChairsSubset/",
#    "/data/dataA/model_interfaces/cts_experts/expert_CTS_add01_TT_flyingChairsSubset/",
#    "/data/dataA/model_interfaces/cts_experts/expert_CTS_add01_SS_flyingChairsSubset/"
#]

model_name = ["PWCWOX1<Chairs>", "PWCWOX1<Things>", "PWCWOX1<Sintel>"]
#model_name = ["CTSExperts_add_chairs", "CTSExperts_add_things", "CTSExperts_add_sintel"]


def plot(image, x, y):
    stretch = (image.shape[1] - 1) / x[-1]
    h = image.shape[0] - 1
    for i in range(1, len(x)):
        rr, cc, val = line_aa(h - int(y[i-1]) - 2, int(x[i-1] * stretch), h - int(y[i]) - 2, int(x[i] * stretch))
        image[rr, cc] = 1 - val


def plot_hist(a, shape, yscale):
    hist, bin_edges = np.histogram(a.ravel(), bins=100)
    hist = hist.astype(np.float) / np.sum(hist)
    image = np.ones(shape)
    plot(image, range(len(hist)), hist * yscale)
    return image

extend = 0
span = 2*extend + 1

out_corr_relu = []
x1 = []
x2 = []
x2_warp = []
flow = []
for model in models:
    out_corr_relu_i, x1_i, x2_i, x2_warp_i, flow_i, l = load_sample(model+"10.npz")
    out_corr_relu.append(out_corr_relu_i)

num_layers = len(out_corr_relu[0])
print(">>", num_layers)

grid = AGrid(
    (span*(len(models)) + 2 * len(models), span*num_layers),
    (out_corr_relu[0][-1].shape[2] + 200, out_corr_relu[0][-1].shape[3] + 200),
    text_params={"font": font},
    title_params={"font": font}, text_height=25)

for l in range(num_layers):
    for m, c in enumerate(out_corr_relu):
        for i in range(81):
            x = i % 9
            y = i // 9

            if (not (4-extend<=x<=4+extend)) or (not (4-extend<=y<=4+extend)):
                continue

            cii = c[-(l+1)][0, i, :, :]
            print(l, cii.shape)
            ci = zoom(cii, 2**(l+1), order=0)
            #ci /= np.linalg.norm(ci, ord=2)
            print("!>", m, l, "|", np.mean(cii), np.std(cii), np.max(cii), cii.shape)

            image = plot_hist(cii, ci.shape, 100)
            #grid.place(-len(models)*2 + m*2, (l * span) + y - 4 + extend, image)
            grid.place(-len(models)*2 + m - 1, (l * span) + y - 4 + extend, image, f"id {m} output")

            p = np.percentile(ci, 50)
            mean = np.mean(ci)

            if True:
                #ci /= p
                ci -= mean
                ci /= np.std(ci) #+ 1e-8
                ci += mean
                #ci += 0.5

            ci = np.clip(ci, 0, 1)
            print("!>", m, l, "|", np.mean(ci), np.std(ci), np.max(ci))
            #print(np.max(ci))

            image = plot_hist(cii, ci.shape, 200)
            grid.place(-len(models) + m - 1, (l * span) + y - 4 + extend, image, f"id {m} normalized")


            grid.place((m*span) + x-4+extend, (l*span) + y-4+extend, ci, model_name[m])

plt.imshow(grid.get_image())
plt.show()