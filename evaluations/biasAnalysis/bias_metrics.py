import math

metric_eval_datasets = [
    "flyingChairsValid",
    "flyingThingsCleanValid",
    "sintelFinalValid",
    "kitti2015Valid"
]

"""
The first values are the best value observed over all pwc or pwcWOX1 baselines.
The second value is the upper median of the pwc and pwcWOX1 baselines.
"""
dataset_performances = {
    "flyingChairsValid": (1.8598, 3.728),
    "flyingThingsCleanValid": (7.432, 13.7999),
    #"sintelFinalValid": (4.6918, 5.8508), <- best is a finetuning result ...
    "sintelFinalValid": (4.9328, 5.8508),
    "kitti2015Valid": (8.1512, 16.3024)
}


def linear_baseline_performance(aepe, dataset_name):
    """
    higher is better:
    1.0: best performance of a baseline
    0.5: average baseline performance
    :param aepe:
    :param dataset_name:
    :return:
    """
    ref_best, ref_median = dataset_performances[dataset_name]
    delta = (ref_median - ref_best)
    return 1.0 - ((aepe - ref_best) / (2*delta))


def normalized_dataset_difference(aepes, measure=linear_baseline_performance):
    normalized_scores = sorted([measure(aepe, dataset_name) for aepe, dataset_name in zip(aepes, metric_eval_datasets)], reverse=True)
    best = normalized_scores[0]
    others = normalized_scores[1:]
    mean_others = sum(others) / len(others)
    return mean_others-best


def cross_dataset_measure(aepes, measure=linear_baseline_performance):
    return 1.0 / (1.0 + math.exp(-(-normalized_dataset_difference(aepes, measure=measure))))


def mean_normalized_performance(aepes, measure=linear_baseline_performance):
    normalized_scores = sorted([measure(aepe, dataset_name) for aepe, dataset_name in zip(aepes, metric_eval_datasets)], reverse=True)
    return sum(normalized_scores) / len(normalized_scores)


def mean_adjusted_normalized_compensated_performance(aepes, measure=linear_baseline_performance):
    normalized_scores = sorted([measure(aepe, dataset_name) for aepe, dataset_name in zip(aepes, metric_eval_datasets)], reverse=True)
    adjusted_scores = [1.0 / (1.0 + math.exp(-(normalized_score - 0.5))) for normalized_score in normalized_scores]
    return sum(adjusted_scores) / len(adjusted_scores)


def inversed_mean_adjusted_normalized_compensated_performance(aepes, measure=linear_baseline_performance):
    score = mean_adjusted_normalized_compensated_performance(aepes, measure=measure)
    return 0.5 + math.log(score / (1 - score))
