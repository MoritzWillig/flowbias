import matplotlib.pyplot as plt

from flowbias.utils.meta_infrastructure import get_eval_summary

evals = get_eval_summary()

models = ["pwcWOX1_on_CTSK_iter", "expertWOX1_CTSK_add01_known_iter"]
iters = ["01", "10", "20", "30", "40", "50", "52", "54", "56", "58", "60"]

x_labels = ["0", "100k", "200k", "300k", "400k", "500k", "600k"]
x_labels_pos = [0, 10, 20, 30, 40, 50, 60]

#flyingChairsValid_lbp	flyingThingsCleanValid_lbp	sintelFinalValid_lbp	kitti2015Valid_lbp

loss_name = "mean_normalized_performance"

xs = list([int(iter) for iter in iters])
ys = []

for model in models:
    y = []
    for iter in iters:
        model_name = model + iter
        loss = float(evals[model_name][loss_name])
        y.append(loss)
    ys.append(y)

plt.figure()
for i, y in enumerate(ys):
    plt.plot(xs, y, label=models[i], marker=".")
plt.xlabel("iteration")
plt.ylabel(loss_name)
plt.ylim(-0.2, 1.0)
plt.xticks(x_labels_pos, x_labels)
plt.show()
