import re
import numpy as np

import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

#results = {key: value for key, value in get_eval_summary().items()}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expert_CTS_add01_..$", key)}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwc_chairs", "I", "H", "pwc_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti", "pwc_chairs", "I", "H", "pwc_kitti", "pwc_on_CTSK"]}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_..$", key)}  # CTSK
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_[CTS][CTS]$", key)}  # CTS(K)
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_split02_..$", key)}  # CTSK
results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_split02_[CTS][CTS]$", key)}  # CTS(K)

#metric_on_X = "normalized_dataset_difference"
#metric_on_Y = "mean_normalized_performance"
#X_title = "normalized_dataset_difference"
#Y_title = "mean_normalized_performance"

#metric_on_X = "middleburyTrain"
#metric_on_Y = "mean_normalized_performance"
#X_title = "middleburyTrain [AEPE]"
#Y_title = "mean_normalized_performance"

metric_on_X = "middleburyTrain"
metric_on_Y = "normalized_dataset_difference"
X_title = "middleburyTrain [AEPE]"
Y_title = "normalized_dataset_difference"


show_iso_lines = False
cut_prefix = True
#title = "Expert models WOX1 CTSK Add01"
title = "Expert models WOX1 CTSK Split02"

plt.figure()
plt.title(title)
xs = np.array([float(value[metric_on_X]) for value in results.values()])
ys = np.array([float(value[metric_on_Y]) for value in results.values()])
is_bases = np.array([value["is_base_model"] == "True" for value in results.values()])
#plt.scatter(xs, ys, c="black")
markers = np.array(["s" if is_base else "." for is_base in is_bases])
plt.scatter(xs[markers == "."], ys[markers == "."], marker=".", c="black")
plt.scatter(xs[markers == "s"], ys[markers == "s"], marker="^", c="black")

print(xs, ys)

for x, y, label in zip(xs, ys, [value["model_id"] for value in results.values()]):
    label = label[-2:] if cut_prefix else label
    plt.annotate(label, (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.xlabel(X_title)
plt.ylabel(Y_title)
#plt.xlim(-0.25, 0.0)
#plt.ylim(0.75, 1.0)


if show_iso_lines:
    x = []
    y = []
    o = 1
    n = 100
    f = 25
    m = n / f
    for i in range(0, n):
        l = i / f
        if i % 2 == 0:
            x.extend([l - o, 0 - o])
            y.extend([0 - o, l - o])
        else:
            x.extend([0 - o, l - o])
            y.extend([l - o, 0 - o])
    for i in range(0, n):
        l = i / f
        if i % 2 == 0:
            x.extend([m - o, l - o])
            y.extend([l - o, m - o])
        else:
            x.extend([l - o, m - o])
            y.extend([m - o, l - o])
    plt.plot(x, y, linestyle=(0, (20, 40)), linewidth=0.2, color="black")


plt.tight_layout()
plt.show()
