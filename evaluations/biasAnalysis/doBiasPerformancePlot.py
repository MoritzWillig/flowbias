import re
import numpy as np

import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

#results = {key: value for key, value in get_eval_summary().items()}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expert_CTS_add01_..$", key)}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwc_chairs", "I", "H", "pwc_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]}
results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti", "pwc_chairs", "I", "H", "pwc_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_..$", key)}

plt.figure()
xs = np.array([float(value["normalized_dataset_difference"]) for value in results.values()])
ys = np.array([float(value["mean_normalized_performance"]) for value in results.values()])
is_bases = np.array([bool(value["is_base_model"]) for value in results.values()])
#plt.scatter(xs, ys, c="black")
markers = np.array(["s" if is_base else "." for is_base in is_bases])
plt.scatter(xs[markers == "."], ys[markers == "."], marker=".", c="black")
plt.scatter(xs[markers == "s"], ys[markers == "s"], marker="^", c="black")

print(xs, ys)

for x, y, label in zip(xs, ys, [value["model_id"] for value in results.values()]):
    plt.annotate(label, (x,y),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center')

plt.xlabel("average drop")
#plt.xlim(-0.25, 0.0)
plt.ylabel("mean normalized performance")
#plt.ylim(0.75, 1.0)


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


plt.show()
