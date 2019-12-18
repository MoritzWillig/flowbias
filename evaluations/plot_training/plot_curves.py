import re
import numpy as np
import matplotlib.pyplot as plt

model_path = "/visinf/home/vimb01/projects/models/A_PWCNet-onChairs-20191121-171532/"

#train
#total_loss_ema=38.7618
train_loss = "total\\_loss\\_ema\\=(.*)\n"

#validation
#epe_avg=3.4301
val_loss = "[^_]epe\\_avg\\=(.*)\n"


with open(model_path+"logbook.txt", "r") as f:
    text = f.read()

train_losses = [float(match) for match in (re.findall(train_loss, text))]
val_losses = [float(match) for match in (re.findall(val_loss, text))]

fig = plt.figure()
ax = plt.subplot(1,1,1)

data = [{
    "x": list(range(len(val_losses))),
    "y": val_losses,
    "label": "avg_epe validation"
}]

for series in data:
    plt.plot(series["x"], series["y"], marker=".", label=series["label"])
    #plt.ylim(np.min(train_losses), np.max(train_losses))

ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
#plt.subplots_adjust(right=0.7)
plt.tight_layout()
plt.show()

#plt.savefig(bbox_inches="tight")