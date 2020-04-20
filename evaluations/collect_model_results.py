import os
import json
from glob import glob
from pathlib import Path

from flowbias.config import Config
from flowbias.evaluations.biasAnalysis.bias_metrics import cross_dataset_measure, metric_eval_datasets, \
    linear_baseline_performance, normalized_dataset_difference, mean_normalized_performance, \
    mean_adjusted_normalized_compensated_performance, inversed_mean_adjusted_normalized_compensated_performance, \
    dataset_performances_outlier_mean, dataset_performances_outlier_span, dataset_performances_median
from flowbias.model_meta import model_meta, model_meta_fields, model_meta_ordering
from flowbias.utils.meta_infrastructure import get_dataset_names

include_valid_only = True
include_cross_dataset_statistics = True
include_dataset = ["middleburyTrain"]

result_file_path = "/data/dataB/meta/full_evals/"

non_dataset_keys = ["model_path", "model_class_name"]

#eval_filenames = sorted(glob(os.path.join(result_file_path, "*")))
evals = {}
#for eval_filename in eval_filenames:
    #eval_name = Path(eval_filename).stem
    #with open(eval_filename, "r") as f:
    #    eval_result = json.loads(f.read())
    #evals[eval_name] = eval_result

for model_name in model_meta_ordering:
    with open(result_file_path+model_name+".json", "r") as f:
        eval_result = json.loads(f.read())
    evals[model_name] = eval_result

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


def dataset_selector(name):
    return (not include_valid_only) or (name.endswith("Valid") or (name in include_dataset))


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


def compute_cross_dataset_measure_linear(eval, return_fields=False):
    if return_fields:
        fields = []
        if include_cross_dataset_statistics:
            #fields.extend(["cross_dataset_measure_linear", *[med+"_lbp" for med in metric_eval_datasets]])
            fields.extend([
                *[med+"_lbp" for med in metric_eval_datasets],
                *[med + "_outlier_median_lbp" for med in metric_eval_datasets],
                *[med + "_outlier_mean_lbp" for med in metric_eval_datasets],
                *[med + "_outlier_span_lbp" for med in metric_eval_datasets]

            ])
        fields.extend([
            "normalized_dataset_difference", "mean_normalized_performance",
            "inversed_mean_adjusted_normalized_compensated_performance",
            "mean_adjusted_normalized_compensated_performance"])
        return fields
    metrics = []

    aepes = [eval[dataset_name]['epe']['average'] for dataset_name in metric_eval_datasets]

    if include_cross_dataset_statistics:
        # for this thesis we are using the lbm_{mean}
        for aepe, dataset_name in zip(aepes, metric_eval_datasets):
            metrics.append(linear_baseline_performance(aepe, dataset_name))
        # different lbm metrics
        for aepe, dataset_name in zip(aepes, metric_eval_datasets):
            metrics.append(linear_baseline_performance(aepe, dataset_name, normalization_values=dataset_performances_median))
        for aepe, dataset_name in zip(aepes, metric_eval_datasets):
            metrics.append(linear_baseline_performance(aepe, dataset_name, normalization_values=dataset_performances_outlier_mean))
        for aepe, dataset_name in zip(aepes, metric_eval_datasets):
            metrics.append(linear_baseline_performance(aepe, dataset_name, normalization_values=dataset_performances_outlier_span))

    metrics.append(normalized_dataset_difference(aepes))
    metrics.append(mean_normalized_performance(aepes))
    metrics.append(inversed_mean_adjusted_normalized_compensated_performance(aepes))
    metrics.append(mean_adjusted_normalized_compensated_performance(aepes))
    return metrics


sorted_datasets = [dataset for dataset in sorted(list(datasets)) if dataset_selector(dataset)]
metric_fields = compute_cross_dataset_measure_linear(None, return_fields=True)

for eval_name, eval in evals.items():
    # infer some results
    compute_chairs_full(eval)
    compute_things_full(eval)
    compute_sintel_clean_full(eval)
    compute_sintel_final_full(eval)
    compute_kitti_full(eval)

    metrics = []
    metrics.extend([str(metric) for metric in compute_cross_dataset_measure_linear(eval)])

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

    line = [eval_name]
    line.extend(results)
    line.extend(metrics)
    line.extend([str(data) for data in model_meta[eval_name]])
    summary[eval_name] = line

#with open(Config.temp_directory+"/eval_summary.json") as file:
#    file.write(json.dumps(summary))

eval_summary_path = Config.eval_summary_path
print(f"writing eval summary to {eval_summary_path}")
with open(eval_summary_path, "w") as file:
    head = ["model_id"]
    head.extend(sorted_datasets)
    head.extend(metric_fields)
    head.extend(model_meta_fields)
    print(">>", head)
    file.write("\t".join(head))
    file.write("\n")

    for model_id in model_meta_ordering:
        line = summary[model_id]
        file.write("\t".join(line))
        file.write("\n")
