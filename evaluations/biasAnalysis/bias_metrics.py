import math

metric_eval_datasets = [
    "flyingChairsValid",
    "flyingThingsCleanValid",
    "sintelFinalValid",
    "kitti2015Valid"
]

"""
The first values are the best value on all baseline or finetuned baselines.
The second value is the upper median of the pwc and pwcWOX1 baselines.
"""
dataset_performances = {
    "flyingChairsValid": (1.8617, 3.728),
    "flyingThingsCleanValid": (7.3425, 13.7999),
    "sintelFinalValid": (4.6918, 5.8508),
    "kitti2015Valid": (11.6487, 17.1193)
}


def linear_baseline_performance(aepe, dataset_name):
    """
    higher is better:
    1.0: best performance seen so far
    0.5: average performance
    :param aepe:
    :param dataset_name:
    :return:
    """
    ref_best, ref_median = dataset_performances[dataset_name]
    delta = (ref_median - ref_best)
    return 1.0 - ((aepe - ref_best) / (2*delta))


def cross_dataset_measure(aepes, measure=linear_baseline_performance):
    r = sorted([measure(aepe, dataset_name) for aepe, dataset_name in zip(aepes, metric_eval_datasets)], reverse=True)
    best = r[0]
    others = r[1:]
    mean_others = sum(others) / len(others)

    return 1.0 / (1.0 + math.exp(-(best-mean_others)))
