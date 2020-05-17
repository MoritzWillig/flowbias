import os
import numpy as np

from flowbias.config import Config

if Config.add_tex_to_path:
    os.environ['PATH'] += os.pathsep + Config.tex_directory

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

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

spacing = 0.5

#X_title = "dataset name"
Y_title = "linear baseline performance"

#bars = ["flyingChairsValid_lbp", "flyingThingsCleanValid_lbp", "sintelFinalValid_lbp", "kitti2015Valid_lbp"]
#bars = ["flyingChairsValid_outlier_mean_lbp", "flyingThingsCleanValid_outlier_mean_lbp", "sintelFinalValid_outlier_mean_lbp", "kitti2015Valid_outlier_mean_lbp"]
bars = ["flyingChairsValid_outlier_span_lbp", "flyingThingsCleanValid_outlier_span_lbp", "sintelFinalValid_outlier_span_lbp", "kitti2015Valid_outlier_span_lbp"]
bar_names = ["Chairs\\textsubscript{Valid}", "Things\\textsubscript{Valid}", "Sintel\\textsubscript{Valid}", "KITTI 2015\\textsubscript{Valid}"]
y = {name: [] for name in bars}

title = "Baseline model distribution (span)"

plt.figure()
plt.title(title)
for i, metric_name in enumerate(bars):
    ys = np.array([float(value[metric_name]) for value in results.values()])
    plt.scatter([i*spacing]*len(ys), ys, marker=".", c="black")

    plt.plot([(i-0.05)*spacing, (i+0.05)*spacing], [0.5, 0.5], color="black", linewidth=0.5)

#plt.xticks([spacing*i for i in range(len(bars))], bar_names, rotation=-45)
plt.xticks([spacing*i for i in range(len(bars))], bar_names)
plt.gcf().autofmt_xdate()

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


#plt.tight_layout()
plt.show()
