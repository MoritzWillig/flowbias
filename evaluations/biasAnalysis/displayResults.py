import csv
import matplotlib.pyplot as plt

from flowbias.config import Config


import numpy as np

resultsPath = Config.temp_directory+"eval_summary.csv"
# model_id model
# flyingChairsFull flyingChairsValid
# flyingThingsCleanFull flyingThingsCleanTrain flyingThingsCleanValid
# kitti2015Full kitti2015Train kitti2015Valid
# sintelCleanFull sintelCleanValid sintelFinalFull sintelFinalValid
# model experiment base_model dataset_base dataset_fine

str_col = {"model_id", "model", "model", "experiment", "base_model", "dataset_base", "dataset_fine"}


def create_equality_filter(col, value):
    def equal_filter(x, return_name=False):
        if return_name:
            return col+":"+value
        return x[col] == value
    return equal_filter

filters = [
    create_equality_filter("model", "pwc"),
    create_equality_filter("experiment", "baseline_single")
    #create_equality_filter("experiment", "fused_conv33")
]

with open(resultsPath, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')

    evals_keys = list(reader.fieldnames)
    evals = {col: [] for col in evals_keys}

    for row in reader:
        if not all([filter(row) for filter in filters]):
            continue

        for key, value in row.items():
            evals[key].append(value if key in str_col else float(value))

fullDatasets = ["flyingChairsFull", "flyingThingsCleanFull", "sintelFinalFull", "kitti2015Full"]

a = []
for i, dataset in enumerate(fullDatasets):
    a.append(evals[dataset])
data = np.array(a).T
print(data.shape)


#data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
#data = data / np.mean(data, axis=0)  # normalize by dataset std


plt.figure()
for i in range(len(fullDatasets)):
    y = data[:, i]
    x = [i] * len(y)
    plt.scatter(x, y)

for i in range(data.shape[0]):
    row = data[i, :]
    x = range(len(row))
    y = row
    plt.plot(
        x, y,
        label=evals["model"][i]+"@"+
              evals["dataset_base"][i]+">"+evals["dataset_fine"][i])

plt.ylim(0, 275)
plt.title(" | ".join([filter(None, return_name=True) for filter in filters]))
#plt.legend(loc='best')
plt.xticks(range(len(fullDatasets)), fullDatasets)
plt.show()