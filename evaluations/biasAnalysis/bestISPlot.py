import re
import numpy as np

import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

#results = {key: value for key, value in get_eval_summary().items()}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expert_CTS_add01_..$", key)}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwc_chairs", "I", "H", "pwc_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if key in ["000", "001", "002", "003", "004", "005"]} #blind fine
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]}
#results = {key: value for key, value in get_eval_summary().items() if key in ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti", "pwc_chairs", "I", "H", "pwc_kitti", "pwc_on_CTSK"]}
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_split02_..$", key)}  # CTSK
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_split02_[CTS][CTS]$", key)}  # CTS(K)
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_..$", key)}  # CTSK
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_[CTS][CTS]$", key)}  # CTS(K)
results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_add01_[CTS][CTS]$", key) or key.startswith("unified")}  # add CTS(K) and unified

#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_linAdd01_..$", key)}  # CTSK
#results = {key: value for key, value in get_eval_summary().items() if re.match("expertWOX1_CTSK_linAdd01_[CTS][CTS]$", key)}  # CTS(K)

perf_metric = "mean_normalized_performance"
bias_metric = "normalized_dataset_difference"

perfs = np.array([float(value[perf_metric]) for value in results.values()])
biass = np.array([float(value[bias_metric]) for value in results.values()])

def interpolatedPerformance(mnp, ad, alpha):
    return (alpha * mnp) + ((1-alpha) * (1+ad))

plt.figure()
#plt.title(title)

eval_range = np.arange(0.0, 1.01, 0.1)
for id, perf, bias in zip(results.keys(), perfs, biass):
    print(">>", id, perf, bias)
    is_range = interpolatedPerformance(perf, bias, eval_range)
    plt.plot(eval_range, is_range)

    plt.text(1.0, is_range[-1], id)


#plt.xlabel(X_title)
#plt.ylabel(Y_title)
plt.xlim(1.0, 0.0)
#plt.ylim(0.75, 1.0)


plt.tight_layout()
plt.show()
