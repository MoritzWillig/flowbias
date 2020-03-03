import argparse

from torch.optim import SGD
import torch
from torch.utils.data.dataloader import DataLoader

from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta, \
    dataset_needs_batch_size_one, get_loss
from flowbias.utils.model_loading import torch_batch_to_cuda

loss_name = "epe"
#dataset_name = "sintelFinalValid" #flyingChairsValid flyingThingsCleanValid sintelFinalValid kitti2015Valid
#datasets = get_available_datasets(datasets="subsets", force_mode="test")
datasets = get_available_datasets(select_by_any_tag=["valid"], exclude_by_tag=["final"], force_mode="test")

#models = ["A", "I", "H", "W", "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]  # base models
models = ["A", "I", "H", "pwc_kitti"]  # base models

# for normalization we require a base model for each dataset
assert(len(models) == len(datasets))


def compute_gradient_score(model, transformer, dataset_name, norm_grads):
    # computes the average sum of gradients
    batch_size = 1
    loss_args = {"args": argparse.Namespace(batch_size=batch_size)}
    loss = get_loss(loss_name, model, dataset_name, loss_args=loss_args).eval().cuda()

    dataset = datasets[dataset_name]
    gpuargs = {"num_workers": 4, "pin_memory": False}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **gpuargs)

    # optimizer = SGD(model.parameters(), lr=1.0)

    # create grads
    grads = {}
    grad_names = []
    for paramName, paramValue, in model.named_parameters():
        grads[paramName] = torch.zeros_like(paramValue).detach()
        grad_names.append(paramName)

    #print(">>", grad_names)

    i = 0
    for sample in loader:
        #optimizer.zero_grad()
        #model.grad.data.zero_()
        #model.zero_grad()

        y = loss(model(torch_batch_to_cuda(transformer(sample))), sample)
        y[loss_name].backward()

        i += 1

    # zero out gradients when moving this into the loop!
    for paramName, paramValue, in model.named_parameters():
        if norm_grads is None:
            grads[paramName] += paramValue.grad.clone().detach() / len(dataset)
        else:
            grads[paramName] += (paramValue.grad.clone().detach() / len(dataset)) #/ norm_grads[paramName]

    #if norm_grads is None:
    #    for paramName, paramValue, in model.named_parameters():
    #        grads[paramName] = grads[paramName].mean()

    avg_grad = 0
    enc_grad = 0
    dec_grad = 0
    context_grad = 0
    avg_grads = []
    for gradName, grad in grads.items():
        g = grad.mean().cpu().numpy()
        avg_grad += g / 98
        avg_grads.append(float(g))

        if gradName.startswith("feature_pyramid_extractor"):
            enc_grad += g / 24
        elif gradName.startswith("flow_estimators"):
            dec_grad += g / 62
        elif gradName.startswith("context_networks"):
            context_grad += g / 12
        else:
            raise Exception("Unknown network component")

    return avg_grad, avg_grads, enc_grad, dec_grad, context_grad, grads


model_datasets = list(datasets.keys())
print(model_datasets)

for i, model_name in enumerate(models):
    model, transformer = load_model_from_meta(model_name)
    model.cuda().eval()
    gradient_score, avg_grads, enc_grad, dec_grad, context_grad, base_grads = compute_gradient_score(model, transformer, model_datasets[i], None)
    print(f"{model_name}@{model_datasets[i]}: {gradient_score}, {enc_grad}, {dec_grad}, {context_grad}")
    print(avg_grads)

    for j, (dataset_name, _) in enumerate(datasets.items()):
        if i == j:
            continue
        model, transformer = load_model_from_meta(model_name)
        model.cuda().eval()

        gradient_score, avg_grads, enc_grad, dec_grad, context_grad, _ = compute_gradient_score(model, transformer, dataset_name, base_grads)
        print(f"{model_name}@{dataset_name}: {gradient_score}, {enc_grad}, {dec_grad}, {context_grad}")
        print(avg_grads)
