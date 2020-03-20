from pathlib import Path
import numpy as np

from flowbias.utils.model_loading import sample_to_torch_batch

from flowbias.utils.meta_infrastructure import load_model_from_meta, get_model_meta, get_available_datasets

recordable_architectures = {
    "PWCNet": "PWCNetRecordable",
    "PWCNetWOX1Connection": "PWCNetWOX1ConnectionRecordable",
    "CTSPWCExpertNetAdd01": "CTSPWCExpertNetAdd01Recordable"
}

#models = ["A", "I", "H", "W"]
#models = ["pwcWOX1_chairs", "pwcWOX1_things", "pwcWOX1_sintel", "pwcWOX1_kitti"]
models = [
    "expert_CTS_add01_CC", "expert_CTS_add01_CT", "expert_CTS_add01_CS",
    "expert_CTS_add01_TC", "expert_CTS_add01_TT", "expert_CTS_add01_TS",
    "expert_CTS_add01_SC", "expert_CTS_add01_ST", "expert_CTS_add01_SS"]

datasets = get_available_datasets(datasets="subsets", force_mode="test")

base_out_path = "/data/dataB/model_interfaces/cts_experts/"

for model_name in models:
    print(f"recording interface for {model_name}")
    for dataset_name, dataset in datasets.items():
        print(f"dataset {dataset_name}")
        sample_interface_path = f"{base_out_path}{model_name}_{dataset_name}/"
        Path(sample_interface_path).mkdir(parents=True, exist_ok=True)

        layer_id = 0
        out_corr_relu_s = {}
        x1_s = {}
        x2_s = {}
        x2_warp_s = {}
        flow_s = {}
        l_s = {}
        data_id = 0


        def clear_sample():
            global out_corr_relu_s, x1_s, x2_s, x2_warp_s, flow_s, l_s, layer_id
            layer_id = 0
            out_corr_relu_s = {}
            x1_s = {}
            x2_s = {}
            x2_warp_s = {}
            flow_s = {}
            l_s = {}


        def recorder_func(out_corr_relu, x1, x2, x2_warp, flow, l):
            global out_corr_relu_s, x1_s, x2_s, x2_warp_s, flow_s, l_s, layer_id
            ct_str = str(layer_id)

            out_corr_relu_s["out_corr_relu_"+ct_str] = out_corr_relu.cpu().data.numpy()
            x1_s["x1_"+ct_str] = x1.data.cpu().numpy()
            x2_s["x2_"+ct_str] = x2.data.cpu().numpy()
            x2_warp_s["x2_warp_"+ct_str] = x2_warp.data.cpu().numpy()
            flow_s["flow_"+ct_str] = flow.data.cpu().numpy()
            l_s["l_"+ct_str] = np.array(l)
            layer_id += 1


        def save_sample():
            global out_corr_relu_s, x1_s, x2_s, x2_warp_s, flow_s, l_s, data_id, sample_interface_path

            np.savez(
                sample_interface_path+str(data_id),
                **out_corr_relu_s,
                **x1_s,
                **x2_s,
                **x2_warp_s,
                **flow_s,
                **l_s)
            data_id += 1


        meta = get_model_meta(model_name)
        recordable_architecture = recordable_architectures[meta.model]
        model, transformer = load_model_from_meta(meta, args={"interface_func" : recorder_func, "args": {}}, force_architecture=recordable_architecture)
        model.eval().cuda()

        for ii in range(len(dataset)):
            clear_sample()
            model(transformer(sample_to_torch_batch(dataset[ii])))
            save_sample()

print("done")
