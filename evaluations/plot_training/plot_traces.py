import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

evals = get_eval_summary()

#models = ["pwcWOX1_on_CTSK_iter", "expertWOX1_CTSK_add01_known_iter"]
models = ["pwcWOX1_on_CTSK_iter", "pwc_on_CTSK_iter"]
iters = ["01", "10", "20", "30", "40", "50", "52", "54", "56", "58", "60"]

#models = ["pwc_every_chairs_", "pwcWOX1_every_chairs_"]
#iters = ["001", "011", "021", "031", "041", "051", "061", "071", "081", "091", "101", "111", "121", "131", "141", "151", "161", "171", "181", "191", "201", "209"]


x_labels = ["0", "100k", "200k", "300k", "400k", "500k", "600k"]
x_labels_pos = [0, 10, 20, 30, 40, 50, 60]
#x_labels_pos = [0, 36, 72, 108, 144, 180, 216]

#flyingChairsValid_lbp flyingThingsCleanValid_lbp sintelFinalValid_lbp kitti2015Valid_lbp

#loss_name = "mean_normalized_performance"
#loss_label = "mean_normalized_performance"
loss_name = "normalized_dataset_difference"
loss_label = "average drop"

xs = list([int(iter) for iter in iters])
ys = []

col = ["dodgerblue", "orange"]

for model in models:
    y = []
    for iter in iters:
        model_name = model + iter
        loss = float(evals[model_name][loss_name])
        y.append(loss)
    ys.append(y)

plt.figure()
for i, y in enumerate(ys):
    plt.plot(xs, y, label=models[i], marker=".", c=col[i])
plt.xlabel("iterations")
plt.ylabel(loss_label)
plt.xlim(min(xs), max(xs))
plt.ylim(-1.0, -0.0)
#plt.ylim(-0.0, 1.0)
plt.xticks(x_labels_pos, x_labels)
plt.show()
