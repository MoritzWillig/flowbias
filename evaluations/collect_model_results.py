import os
import json
from glob import glob
from pathlib import Path

from flowbias.config import Config
from flowbias.model_meta import model_meta, model_meta_fields, model_meta_ordering
from flowbias.utils.meta_infrastructure import get_dataset_names

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
datasets = set(get_dataset_names())
# add datasets that may be specific for some models
for eval in evals.values():
    datasets = datasets.union(set(eval.keys()))
datasets = datasets.difference(set(non_dataset_keys))


summary = {}

print("found", len(evals), "evaluations")
print("datasets:", datasets)
print("datasets:", next(iter(evals.values())).keys())


def compute_dataset_full(eval, train_name, valid_name, full_name, train_size, valid_size):
    # full is already computed
    if full_name in eval:
        return
    # train and valid splits have to be present
    if (train_name not in eval) or (valid_name not in eval):
        return

    train_dataset_ratio = train_size / (train_size + valid_size)
    valid_dataset_ratio = valid_size / (train_size + valid_size)
    dataset_full = (eval[train_name]["epe"]["average"] * train_dataset_ratio) + \
                   (eval[valid_name]["epe"]["average"] * valid_dataset_ratio)
    eval[full_name] = {
        "epe": {
            "average": dataset_full,
            "min": min(eval[train_name]["epe"]["min"], eval[valid_name]["epe"]["min"]),
            "max": max(eval[train_name]["epe"]["max"], eval[valid_name]["epe"]["max"])
        }}


def compute_chairs_full(eval):
    compute_dataset_full(eval, "flyingChairsTrain", "flyingChairsValid", "flyingChairsFull", 22232, 640)


def compute_things_full(eval):
    compute_dataset_full(eval, "flyingThingsCleanTrain", "flyingThingsCleanValid", "flyingThingsCleanFull", 19635, 3823)


def compute_sintel_clean_full(eval):
    compute_dataset_full(eval, "sintelCleanTrain", "sintelCleanValid", "sintelCleanFull", 908, 133)


def compute_sintel_final_full(eval):
    compute_dataset_full(eval, "sintelFinalTrain", "sintelFinalValid", "sintelFinalFull", 908, 133)


def compute_kitti_full(eval):
    compute_dataset_full(eval, "kitti2015Train", "kitti2015Valid", "kitti2015Full", 160, 40)


sorted_datasets = sorted(list(datasets))

for eval_name, eval in evals.items():
    # infer some results
    compute_chairs_full(eval)
    compute_things_full(eval)
    compute_sintel_clean_full(eval)
    compute_sintel_final_full(eval)
    compute_kitti_full(eval)

    results = []
    missing = []
    for dataset in sorted_datasets:
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
    head.extend(sorted_datasets)
    head.extend(model_meta_fields)
    print(">>", head)
    file.write("\t".join(head))
    file.write("\n")

    for model_id in model_meta_ordering:
        line = summary[model_id]
        file.write("\t".join(line))
        file.write("\n")
