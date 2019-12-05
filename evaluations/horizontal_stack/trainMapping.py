import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configuration import _generate_trainable_params
from datasets.pwcInterfaceDataset import PWCInterfaceDataset
from models.pwc_modules import initialize_msra

sample_interface_pathA = "/data/vimb01/experiments/pwcinterfaceA/*"
sample_interface_pathB = "/data/vimb01/experiments/pwcinterfaceB/*"
connector_layers = 1
connector_kernel_size = 1
# pwc net has multiple layers. A connector has to be trained for each layer:
train_level = 0
batch_size = 8

epochs = 10
print_every = 100

gpuargs = {}


# create dataset
def create_dataset(shuffle=True):
    dataset = PWCInterfaceDataset({}, sample_interface_pathA, sample_interface_pathB, train_level)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **gpuargs)
    return dataset, loader


dataset, loader = create_dataset()
demo_out_corr_relu, demo_x1, demo_flow, demo_l = dataset[0]

# compute feature count
out_corr_layers = demo_flow.shape[0]
x1_layers = demo_flow.shape[0]
flow_layers = demo_flow.shape[0]
total_layers = out_corr_layers + x1_layers + flow_layers




# train
net = PWCConnectorNet({})
loss = nn.L1Loss()
optimizer = optim.Adam(_generate_trainable_params(net), lr=1e-4, weight_decay=4e-4)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every-1:    # print every 2000 mini-batches
            print(f'[${epoch + 1}, ${i+1,5}] loss: ${running_loss / print_every,.3}')
            running_loss = 0.0

print('Finished Training')
