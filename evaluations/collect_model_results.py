import os
import json
from glob import glob
from pathlib import Path

from flowbias.config import Config
from flowbias.model_meta import model_meta, model_meta_fields, model_meta_ordering

result_file_path = "/data/dataB/meta/full_evals/"

non_dataset_keys = ["model_path", "model_class_name"]

eval_filenames = sorted(glob(os.path.join(result_file_path, "*")))
evals = {}
for eval_filename in eval_filenames:
    eval_name = Path(eval_filename).stem
    with open(eval_filename, "r") as f:
        eval_result = json.loads(f.read())

    evals[eval_name] = eval_result

# extract all known datasets
datasets = set()
for eval in evals.values():
    datasets = datasets.union(set(eval.keys()))
datasets = datasets.difference(set(non_dataset_keys))


summary = {}

print("found", len(evals), "evaluations")
print("datasets:", datasets)
print("datasets:", next(iter(evals.values())).keys())

for eval_name, eval in evals.items():
    results = []
    missing = []
    for dataset in datasets:
        if dataset in eval:
            results.append(f"{eval[dataset]['epe']['average']:.4f}")
        else:
            results.append("None")
            missing.append(dataset)
    if len(missing) != 0:
        print(eval_name, "missing results for", missing)

    line = [eval_name, eval["model_class_name"]]
    line.extend(results)
    line.extend(["None" if data is None else data for data in model_meta[eval_name]])
    summary[eval_name] = line

#with open(Config.temp_directory+"/eval_summary.json") as file:
#    file.write(json.dumps(summary))

with open(Config.temp_directory+"/eval_summary.csv", "w") as file:
    head = ["model_id", "model"]
    head.extend(datasets)
    head.extend(model_meta_fields)
    print(">>", head)
    file.write("\t".join(head))
    file.write("\n")

    for model_id in model_meta_ordering:
        line = summary[model_id]
        file.write("\t".join(line))
        file.write("\n")
