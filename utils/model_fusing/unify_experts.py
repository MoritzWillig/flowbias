import torch

from flowbias.utils.meta_infrastructure import load_model_from_meta
from flowbias.utils.model_loading import save_model

resulting_model_dir = "/data/dataB/unifiedModelsExpert/"

model_name = "expertWOX1_CTSK_add01_expert0"
fused_name = "unifiedCTS_avg_expertWOX1_CTSK_add01_expert"
num_expected_experts = 4
exclude_experts = [3]

with torch.no_grad():
    model, _ = load_model_from_meta(model_name)

    expert_params = {}
    for param_name, param in model.named_parameters():
        if "expert" not in param_name:
            continue

        if param_name.endswith(".0.weight"):
            t = -10
        elif param_name.endswith(".0.bias"):
            t = -8
        else:
            raise RuntimeError(f"unknown parameter type {param_name}")

        # context networks encodes <EXPERTID.x.x>
        if (param_name[t-1] == "." ) and (param_name[t-2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
            t += -2

        #cut out expert id
        param_base_name = param_name[:t] + param_name[(t+1):]
        print(param_base_name, param_name)

        expert_id = param_name[t]
        if int(expert_id) in exclude_experts:
            continue

        if param_base_name not in expert_params:
            param_data = {
                "experts": [],
                "sample_full": param_name
            }
            expert_params[param_base_name] = param_data
        else:
            param_data = expert_params[param_base_name]

        param_data["experts"].append(param)

    for base_name, param_data in expert_params.items():
        if len(param_data["experts"]) != num_expected_experts - len(exclude_experts):
            sample_full = param_data["sample_full"]
            num_experts = len(param_data["experts"])
            raise RuntimeError(f"found base name with {num_experts} experts (expected {num_expected_experts - len(exclude_experts)} experts): {base_name} [full: {sample_full}]")

        # average expert parameters
        s = torch.zeros_like(param_data["experts"][0].data)
        for param in param_data["experts"]:
            s += param.data
        s /= len(param_data["experts"])

        for param in param_data["experts"]:
            param.copy_(s)

    save_model(model, resulting_model_dir + fused_name + "/")

print("done")
