import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import re

paths = [
    "/data/dataA/temp/activations/expertWOX1_CTSK_add01_expert0_middleburyTrain_4/numpy/",
    #"/data/dataA/temp/activations/expertWOX1_CTSK_add01_expert1_middleburyTrain_4/numpy/",
    #"/data/dataA/temp/activations/expertWOX1_CTSK_add01_expert2_middleburyTrain_4/numpy/"
    "/data/dataA/temp/activations/expertWOX1_CTSK_add01_expert3_middleburyTrain_4/numpy/"
    ]

exp_ids_path = "/data/dataA/temp/activations/expertWOX1_CTSK_add01_expert0_middleburyTrain_4/"


#paths = [
#    "/data/dataA/temp/activations/expertWOX1_CTSK_linAdd01_expert0_middleburyTrain_4/numpy/",
#    "/data/dataA/temp/activations/expertWOX1_CTSK_linAdd01_expert1_middleburyTrain_4/numpy/",
#    "/data/dataA/temp/activations/expertWOX1_CTSK_linAdd01_expert2_middleburyTrain_4/numpy/"
#    ]

#exp_ids_path = "/data/dataA/temp/activations/expertWOX1_CTSK_linAdd01_expert0_middleburyTrain_4/"


filepaths = sorted(glob.glob(paths[0]+"*"))

variances = dict()
max_id = -1
for filepath in filepaths:
    filename = os.path.basename(filepath)

    m = re.match("(^\d*)", filename)
    layer_id = int(m.group(1))
    if layer_id > max_id:
        max_id = layer_id

    activations = [np.load(path+filename) for path in paths]

    full_activations = np.zeros((len(activations), activations[0].size))

    for i, activation in enumerate(activations):
        full_activations[i, :] = activation.flat

    #print(">>", np.var(full_activations, axis=0).shape)

    #print(filename)
    #print(np.mean(full_activations, axis=1))
    variance = np.mean(np.var(full_activations, axis=0))
    #variance = (np.mean(full_activations)) #plots the mean instead of variance
    #print(variance)
    variances[layer_id] = variance

    #print(activations)

isExpert = dict()
for path in sorted(glob.glob(exp_ids_path+"*")):
    if not os.path.isfile(path):
        continue
    layer_id = int(re.match("(^\d*)", os.path.basename(path)).group(1))
    if layer_id not in variances:
        continue
    isExpert[layer_id] = "xpert" in os.path.basename(path)

plt.figure()

ib=0
ie=0
#base_var = [1000] * (len(variances) // 3)
base_var = []
#expert_var = [1000] * (len(variances) // 3)
expert_var = []

last_was_exp = not isExpert[0]
for i in range(max_id):
    if i not in variances:
        continue
    var = variances[i]
    expert = isExpert[i]

    if expert:
        if not last_was_exp:
            expert_var.append(var)

        # min_value is 1 or -1 and inverses the comparison
        # we want to record the min variance (since the variance after relu is the smallest) and the max mean
        if abs(expert_var[-1]) < abs(var):
            expert_var[-1] = var
        #ie+=1
        #ib +=1
    else:
        #we just switched to base
        if last_was_exp:
            base_var.append(var)

        if abs(base_var[-1]) < abs(var):
            base_var[-1] = var
        #ib += 1

    last_was_exp = expert

for i, (bv, ev) in enumerate(zip(base_var, expert_var)):
    z = 0 if bv>ev else 1
    if bv < 0:
        z = 1 - z
    #plt.plot([i, i], [0, bv], color="black", zorder=z)
    #plt.plot([i, i], [0, ev], color="orange", zorder=1-z)

    plt.bar(i, bv, width=0.5, color="black", zorder=z)
    plt.bar(i, ev, width=0.5, color="orange", zorder=1 - z)

plt.xticks([])
#plt.xticks([0, 12, 24, 54, 60], ["input", "feature extractor", "feature extractor", "flow estimator", "output"])

#id//3//2
#0..72 pyramidextr A
#73..145 pyramidextr B
#145..329 flow estim
#330..370 context network

plt.show()

print(f"{len(base_var)} base; {len(expert_var)} experts")
print(">>", np.mean(base_var), np.mean(expert_var))

