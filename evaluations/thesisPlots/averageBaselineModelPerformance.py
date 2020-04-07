import re
import numpy as np

import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

# baselines
results = {key: value for key, value in get_eval_summary().items() if key in [
    "pwc_chairs", "I", "H", "pwc_kitti",
    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]}

# baselines + finetuned
#results = {key: value for key, value in get_eval_summary().items() if key in [
#    "pwc_chairs", "I", "H", "pwc_kitti",
#    "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti",
#    "F","K","R","S","V","Y","X","O"]}



#X_title = "dataset name"
Y_title = "linear baseline performance"

bars = ["flyingChairsValid_lbp", "flyingThingsCleanValid_lbp", "sintelFinalValid_lbp", "kitti2015Valid_lbp"]
y = {name: [] for name in bars}

title = "Baseline model distribution"

plt.figure()
plt.title(title)
for i, metric_name in enumerate(bars):
    ys = np.array([float(value[metric_name]) for value in results.values()])
    plt.scatter([i]*len(ys), ys, marker=".", c="black")

#for x, y, label in zip(xs, ys, [value["model_id"] for value in results.values()]):
#    label = label[-2:] if cut_prefix else label
#    plt.annotate(label, (x,y),
#                 textcoords="offset points",
#                 xytext=(0,10),
#                 ha='center')

#plt.xlabel(X_title)
plt.ylabel(Y_title)
#plt.xlim(-0.25, 0.0)
plt.ylim(0.0, 1.05)


plt.tight_layout()
plt.show()
