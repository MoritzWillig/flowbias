import numpy as np


class SeriesStatistic(object):

    def __init__(self):
        self.series = []
        self.weights = []

    def push_value(self, value, weight=0.0):
        self.series.append(value)
        self.weights.append(weight)

    def print_statistics(self):
        s = np.array(self.series)
        print(f"average: {np.average(s)}")
        print(f"min: {np.min(s)}")
        print(f"max: {np.max(s)}")
        print(s)

    def get_statistics(self, report_individual_values=True):
        s = np.array(self.series).astype(float)
        statistics = {
            "average": np.average(s, weights=self.weights),
            "min": np.min(s),
            "max": np.max(s),
        }
        if report_individual_values:
            statistics["values"] = list(s)

        return statistics
