from torch.optim import SGD
import torch
from torch.utils.data.dataloader import DataLoader

from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta, \
    dataset_needs_batch_size_one, get_loss

loss_name = "epe"
#dataset_name = "sintelFinalValid" #flyingChairsValid flyingThingsCleanValid sintelFinalValid kitti2015Valid
datasets = get_available_datasets(datasets="subsets", force_mode="test")

models = ["A", "I", "H", "W", "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]  # base models


def compute_gradient_score(model, transformer, dataset_name):
    # computes the average sum of gradients
    batch_size = 1
    loss_args = { "batch_size": batch_size }
    loss = get_loss(loss_name, model, dataset_name, loss_args=loss_args).cuda()

    gpuargs = {"num_workers": 4, "pin_memory": False}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **gpuargs)

    optimizer = SGD(model.parameters(), lr=1.0)

    # create grads
    grads = {}
    grad_names = []
    for paramName, paramValue, in models.named_parameters():
        grads[paramName] = torch.zeros_like(paramValue)
        grad_names.append(paramName)

    print(">>", grad_names)

    i=0
    for sample in loader:
        #optimizer.zero_grad()
        model.grad.data.zero_()

        y = loss(model(transformer(sample)))
        y.backward()

        i+=1

    for paramName, paramValue, in models.named_parameters():
        grads[paramName] += paramValue.grad.clone() / len(dataset)

    avg_grad = 0
    for gradName, grad in grads.items():
        avg_grad += grad.cpu().numpy().average()

    return avg_grad


for model_name in models:
    for dataset_name, dataset in datasets.items():
        model, transformer = load_model_from_meta(model_name)
        model.cuda().eval()

    gradient_score = compute_gradient_score(model, transformer, dataset_name)
    print(f"{model_name}@{dataset_name}: {gradient_score}")
