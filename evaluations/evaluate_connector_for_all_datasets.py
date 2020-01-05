from flowbias.datasets.pwcInterfaceDataset import PWCInterfaceDatasetValid
from flowbias.losses import MSEConnectorLoss
from flowbias.models import PWCTrainableConvConnector33
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch


class ValArgs:
    def __init__(self):
        self.batch_size = 1


interfaces_roots = {
    "ac": "/data/dataA/model_interfaces/A_chairs",
    "ak": "/data/dataA/model_interfaces/A_kitti",
    "as": "/data/dataA/model_interfaces/A_sintel",
    "at": "/data/dataA/model_interfaces/A_things",
    "hc": "/data/dataA/model_interfaces/H_chairs",
    "hk": "/data/dataA/model_interfaces/H_kitti",
    "hs": "/data/dataA/model_interfaces/H_sintel",
    "ht": "/data/dataA/model_interfaces/H_things",
    "ic": "/data/dataA/model_interfaces/I_chairs",
    "ik": "/data/dataA/model_interfaces/I_kitti",
    "is": "/data/dataA/model_interfaces/I_sintel",
    "it": "/data/dataA/model_interfaces/I_things",
    "wc": "/data/dataA/model_interfaces/W_chairs",
    "wk": "/data/dataA/model_interfaces/W_kitti",
    "ws": "/data/dataA/model_interfaces/W_sintel",
    "wt": "/data/dataA/model_interfaces/W_things"
}

group_order = ["c", "k", "s", "t"]
learned_connector_groups = {
    "ah": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_chairs-20200104-025720/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_kitti-20200104-031002/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_sintel-20200104-034310/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AH_33_things-20200104-040603/checkpoint_best.ckpt"],
    "ai": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AI_33_chairs-20200104-043146/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AI_33_kitti-20200104-044433/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AI_33_sintel-20200104-051659/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AI_33_things-20200104-053915/checkpoint_best.ckpt"],
    "aw": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AW_33_chairs-20200104-060453/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AW_33_kitti-20200104-061738/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AW_33_sintel-20200104-065019/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-AW_33_things-20200104-071236/checkpoint_best.ckpt"],
    "ha": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HA_33_chairs-20200104-073813/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HA_33_kitti-20200104-075103/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HA_33_sintel-20200104-082356/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HA_33_things-20200104-084614/checkpoint_best.ckpt"],
    "hi": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HI_33_chairs-20200104-091203/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HI_33_kitti-20200104-092456/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HI_33_sintel-20200104-095806/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HI_33_things-20200104-102203/checkpoint_best.ckpt"],
    "hw": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HW_33_chairs-20200104-105057/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HW_33_kitti-20200104-110357/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HW_33_sintel-20200104-113739/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-HW_33_things-20200104-115955/checkpoint_best.ckpt"],
    "ia": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IA_33_chairs-20200104-122728/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IA_33_kitti-20200104-124039/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IA_33_sintel-20200104-131411/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IA_33_things-20200104-133703/checkpoint_best.ckpt"],
    "ih": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IH_33_chairs-20200104-140356/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IH_33_kitti-20200104-141659/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IH_33_sintel-20200104-145030/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IH_33_things-20200104-151254/checkpoint_best.ckpt"],
    "iw": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IW_33_chairs-20200104-154024/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IW_33_kitti-20200104-155318/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IW_33_sintel-20200104-162701/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-IW_33_things-20200104-165052/checkpoint_best.ckpt"],
    "wa": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WA_33_chairs-20200104-172156/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WA_33_kitti-20200104-173541/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WA_33_sintel-20200104-181111/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WA_33_things-20200104-183550/checkpoint_best.ckpt"],
    "wh": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WH_33_chairs-20200104-195000/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WH_33_kitti-20200104-200412/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WH_33_sintel-20200104-203925/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WH_33_things-20200104-210348/checkpoint_best.ckpt"],
    "wi": [
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WI_33_chairs-20200104-221423/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WI_33_kitti-20200104-222755/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WI_33_sintel-20200104-230204/checkpoint_best.ckpt",
        "/data/dataB/models/0_connectors33/PWCTrainableConvConnector33-WI_33_things-20200104-232516/checkpoint_best.ckpt"]
}

for connected_networks, learned_connector_group in learned_connector_groups.items():
    for dataset_name, learned_connector in zip(group_order, learned_connector_group):
        model = PWCTrainableConvConnector33(ValArgs) # we use the trainable connector, since it processes the samples directly
        load_model_parameters(model, learned_connector)
        model.cuda()
        loss = MSEConnectorLoss(ValArgs()).cuda()
        model.eval()
        loss.eval()

        total_error = 0
        # evaluate mapping between datasets
        for g in group_order:
            ac = connected_networks[0] + g
            bc = connected_networks[1] + g
            rootA = interfaces_roots[ac]
            rootB = interfaces_roots[bc]

            dataset = PWCInterfaceDatasetValid(ValArgs(), rootA, rootB)

            error = 0
            for i in range(len(dataset)):
                sample = sample_to_torch_batch(dataset[i])
                error += float(loss(model(sample), sample)["total_loss"].cpu().detach().numpy())

            print(f"evaluated {connected_networks}{dataset_name} for {ac}-{bc}: {error}")
            total_error += error
        print(f"{connected_networks}{dataset_name} summed total error: {total_error}")

print("finished")
