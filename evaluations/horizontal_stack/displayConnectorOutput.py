from flowbias.datasets import PWCInterfaceDatasetValid
from flowbias.models import PWCConvAppliedConnector
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

rootA = "/data/dataA/model_interfaces/pwc/A_chairs/"
rootB = "/data/dataA/model_interfaces/pwc/H_chairs/"
#connector_path = "/data/dataB/models/0_connectors13/PWCTrainableConvConnector13-AH_13_chairs-20200101-160051/checkpoint_best.ckpt"
connector_path = "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_chairs-20200104-025720/checkpoint_best.ckpt"
#connector_path = "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_sintel-20200104-034310/checkpoint_best.ckpt"


def shift(a, vmin, vmax):
    return (a + vmin) / (vmax - vmin)


def norm(a):
    m = np.mean(a, (0,1))
    am = a - m
    s=np.std(am, (0,1))
    print(m.shape, am.shape, s.shape)
    return am / (2*s)[None, None, :]


connector = PWCConvAppliedConnector({}, 3, 3)
load_model_parameters(connector, connector_path)
connector.cuda()

dataset = PWCInterfaceDatasetValid({}, rootA, rootB)


sample = sample_to_torch_batch(dataset[0])
print(sample.keys())

results = []
expected = []
inputs = []
for l in range(5):
    results.append(connector.forward(sample[f"input_x1_{l}"], l).cpu().detach().numpy())
    expected.append(sample[f"target_x1_{l}"].cpu().detach().numpy())
    inputs.append(sample[f"input_x1_{l}"].cpu().detach().numpy())

print(results[4].shape)
print(np.min(results[4]), np.max(results[4]))
print(np.min(expected[4]), np.max(expected[4]))

ra = 15
rb = 18

dpi=72
for l in range(5):
    f = 5/(l+1)
    plt.figure(figsize=(200*f/dpi, 500*f/dpi))

    plt.subplot(311)
    plt.imshow(norm(np.transpose(inputs[l][0, ra:rb, :, :], [1, 2, 0])))
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplot(312)
    plt.imshow(norm(np.transpose(results[l][0, ra:rb, :, :], [1, 2, 0])))
    plt.gca().set_axis_off()
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.subplot(313)
    plt.imshow(norm(np.transpose(expected[l][0, ra:rb, :, :], [1, 2, 0])))
    plt.gca().set_axis_off()
    plt.subplots_adjust(wspace=None, hspace=None)

    #plt.show()
    plt.savefig(
        f"/data/dataB/temp/connector_output_{l}.png",
        transparent=True,
        bbox_inches='tight',
        pad_inches=0,
        dpi=dpi)
