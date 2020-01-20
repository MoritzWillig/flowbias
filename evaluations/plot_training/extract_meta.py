import re
import json
import matplotlib.pyplot as plt

from .dataset_meta import models

results_path = "/home/moritz/projects/flow_thesis/results/"



for i in range(len(models)):
    meta_data = models[i]

    if ("ignore" in meta_data) and (meta_data["ignore"]):
        continue

    with open(meta_data["path"]+"logbook.txt") as file:
        data_str = file.read()

    train_results = [float(i) for i in re.findall(r"total\_loss\_ema\=([0-9\.]+)", data_str)]
    val_results = [float(i) for i in re.findall(r"[^_]epe\_avg\=([0-9\.]+)", data_str)]

    plt.plot(range(len(train_results)), train_results, label=str(i))

    results = {
        "train": {
            "epe": train_results
        },
        "val": {
            "epe": val_results
        }
    }

    serialized = json.dumps(results)
    with open(results_path + meta_data["name"] + ".json", "w+") as file:
        file.write(serialized)

plt.show()
