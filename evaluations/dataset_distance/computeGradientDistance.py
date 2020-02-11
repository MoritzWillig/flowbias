
from flowbias.utils.meta_infrastructure import get_available_datasets, load_model_from_meta, \
    dataset_needs_batch_size_one, get_loss

loss_name = "epe"
dataset_name = "sintelFinalValid" #flyingChairsValid flyingThingsCleanValid sintelFinalValid kitti2015Valid

models = ["A", "I", "H", "W", "pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]  # base models


def compute_gradient_score(model, transformer, dataset_name):
    # computes the average sum of gradients ...
    datasets = get_available_datasets(restrict_to=[dataset_name], force_mode="test")
    dataset = datasets[dataset_name]

    loss_args = {
        "batch_size": 1 if dataset_needs_batch_size_one(dataset_name) else 8
    }
    loss = get_loss(loss_name, model, dataset_name, loss_args=loss_args).cuda()

    # optimizer = optim.SGD(model.parameters(), lr=1.0)
    for i, sample in enumerate(dataset):
        if i == 100:
            break

        # optimizer.zero_grad()
        model.grad.data.zero_()

        y = loss(model(transformer(sample)))
        y.backward()

        summed += model.grad


for model_name in models:
    model, transformer = load_model_from_meta(model_name)
    model.cuda().eval()

    gradient_score = compute_gradient_score(model, transformer, dataset_name)
    print(f"{gradient_score}/{dataset_name}: {gradient_score}")
