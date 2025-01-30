import torch
from data_utils import *


def aggregate_models_normal(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_ffa(global_model, client_models):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_ours(global_model, client_models, args):

    global_model = (
        global_model.to("cuda") if torch.cuda.is_available() else global_model
    )
    global_dict = global_model.state_dict()

    for k in global_dict.keys():

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            ).mean(0)

    for client_model in client_models:
        state_dict = client_model.state_dict()
        for k in global_dict.keys():

            if "classifier" in k:
                state_dict[k] = global_dict[k]
        client_model.load_state_dict(state_dict)

    for name, module in global_model.named_modules():

        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):

            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [client_model.state_dict()[lora_A_keys].detach() for client_model in client_models]
            )
            lora_B_weights = torch.stack(
                [client_model.state_dict()[lora_B_keys].detach() for client_model in client_models]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_models))
            ) / len(client_models)

            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = (
                args.lora_alpha / np.sqrt(args.lora_r)
                if args.rslora
                else args.lora_alpha / args.lora_r
            )

            residue = M - lora_B_avg @ lora_A_avg

            global_dict[lora_A_keys] = lora_A_avg
            global_dict[lora_B_keys] = lora_B_avg
            global_dict[base_layer_keys] += torch.transpose(
                residue * scaling_factor, 1, 0
            )

    global_model.load_state_dict(global_dict)

    return global_model


def aggregate_models_flora(global_model, client_models):
    """
    Concat the LoRA_A at dim 0 and LoRA_B at dim 1,
    then calculate delta_W = concat_B @ concat_A and update it to the base layer weight
    :param global_model:
    :param client_models:
    :return:
    """
    global_dict = global_model.state_dict()
    lora_dict = {}
    for k in global_dict.keys():
        name = '.'.join(k.split('.')[:-3]) + '.base_layer.weight'
        if "lora_A" in k:  # Only aggregate LoRA parameters
            lora_dict[name] = torch.concat(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0
            )
        if "lora_B" in k:
            B_concat = torch.concat(
                [client_models[i].state_dict()[k].float() for i in range(len(client_models))], 1
            )
            lora_dict[name] = B_concat @ lora_dict[name]
    for name, delta_w in lora_dict.items():
        global_dict[name] += delta_w
    global_model.load_state_dict(global_dict)

    return global_model
