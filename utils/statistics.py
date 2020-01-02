import numpy as np


class SeriesStatistic(object):

    def __init__(self):
        self.series = []

    def push_value(self, value):
        self.series.append(value)

    def print_statistics(self):
        s = np.array(self.series)
        print(f"average: {np.average(s)}")
        print(f"min: {np.min(s)}")
        print(f"max: {np.max(s)}")
        print(s)

    def get_statistics(self):
        s = np.array(self.series).astype(float)
        return {
            "average": np.average(s),
            "min": np.min(s),
            "max": np.max(s),
            "values": list(s)
        }
