import os
import time

os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import json
import warnings
from copy import deepcopy
import numpy as np
import torch

import wandb
from unsloth import FastLanguageModel
from data_utils import create_dolly_data, create_client_dataloaders_nlg
from fed_agg import aggregate_models_normal, aggregate_models_ffa, aggregate_models_flora
from models import create_peft_llama_model_dolly, create_peft_llama_model_dolly_ffa, create_llama3_base_model
from train_eval import evaluate_instruction_tuning_save_text, train_client_dolly

parser = argparse.ArgumentParser(description="Federated Instruction tuning with LoRA")

parser.add_argument("--agg_type", type=str, default="normal", help="Type of aggregation")
parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
parser.add_argument("--local_epochs", type=int, default=2, help="Number of local epochs")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha value")
parser.add_argument("--lora_dropout", type=float, default=0., help="LoRA dropout value")
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--load_in_4bit", action="store_true", help="Quantized LoRA")
parser.add_argument("--gradient_checkpointing", action="store_true", help="gradient checkpointing")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
parser.add_argument("--idx", type=int, default=0, help="Index of the save folder")
parser.add_argument("--log", action="store_true", help="Log the results")
parser.add_argument("--run_dir", type=str, default='fed_dolly_output', help="Directory to store logs")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulations")
parser.add_argument("--lr_scheduler_type", type=str, default='linear', help="lr_scheduler_type")
parser.add_argument("--optim", type=str, default="adamw_8bit", help="optimizer type")
parser.add_argument("--dtype", default=torch.float16)


args = parser.parse_args()

wandb.init(project="fed_train_dolly_2", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

warnings.filterwarnings("ignore")


def get_next_run_number(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1

    existing_runs = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    return max(existing_runs, default=0) + 1


def save_args(args, directory):
    args_file = os.path.join(directory, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)


def federated_learning(task):
    start = time.time()
    # Load model and Llama tokenizer
    base_model, tokenizer = create_llama3_base_model(args)
    train_data, test_data, tokenizer = create_dolly_data(args, tokenizer)
    client_data = create_client_dataloaders_nlg(train_data, args)

    if args.agg_type == "ffa":
        model = create_peft_llama_model_dolly_ffa(args, base_model)
    else:
        model = create_peft_llama_model_dolly(args, base_model)

    for r in range(args.rounds):
        print(f"Round {r + 1}/{args.rounds}")
        model = FastLanguageModel.for_training(model)

        if args.agg_type == "ffa":
            global_state = deepcopy({k: deepcopy(v) for k, v in model.state_dict().items() if "lora_B" in k or "classifier" in k})
        elif args.agg_type == "normal":
            model = create_peft_llama_model_dolly(args, base_model)
            global_state = deepcopy({k: deepcopy(v) for k, v in model.state_dict().items() if "lora" in k or "classifier" in k})
        elif args.agg_type == "flora":
            global_state = deepcopy({k: deepcopy(v) for k, v in model.state_dict().items() if ".base_layer.weight" in k or "classifier" in k})
        else:
            raise Exception("Wrong agg type")

        # Train on selected clients
        client_models = []
        for client in range(args.num_clients):
            print(f'Client {client}/ {args.num_clients}')
            # print('Start :', global_state[k])
            if args.agg_type == "ffa":
                model = create_peft_llama_model_dolly_ffa(args, base_model)
                model.load_state_dict(global_state, strict=False)
            elif args.agg_type == "normal":
                model = create_peft_llama_model_dolly(args, base_model)
                model.load_state_dict(global_state, strict=False)
            elif args.agg_type == "flora":
                model = create_peft_llama_model_dolly(args, base_model)
                model.load_state_dict(global_state)
            else:
                raise Exception('agg_type only support ffa flora normal.')
            # print('Before train', model.state_dict()[k], global_state[k])
            client_model = train_client_dolly(
                model, tokenizer, client_data[client], args#, val_data
            )
            # print('After train', client_model.state_dict()[k], global_state[k])
            client_models.append({k: deepcopy(v) for k, v in client_model.state_dict().items() if "lora" in k or "classifier" in k})

        if args.agg_type == "normal":
            global_model, base_model = aggregate_models_normal(model, client_models, base_model)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(model, client_models)
        elif args.agg_type == 'flora':
            global_model = aggregate_models_flora(model, client_models)
        else:
            raise Exception('No agg_type')
        # print('After agg:', client_models[0][k], global_model.state_dict()[k])
        if args.log:
            base_dir = "text_store_new/" + args.agg_type
            run_number = get_next_run_number(base_dir)
            run_dir = os.path.join(base_dir, str(run_number))
            os.makedirs(run_dir)
            save_args(args, run_dir)
            args.run_dir = run_dir
        global_model = FastLanguageModel.for_inference(global_model)
        total_scores = evaluate_instruction_tuning_save_text(global_model, test_data[:200], tokenizer, args)
        # print('After evaluation', client_models[0][k], global_model.state_dict()[k])
        wandb.log({'Rouge-L score': total_scores["ROUGE_L"], 'BLEU score': total_scores["Bleu"]})
    print(time.time()-start)


# Main execution
if __name__ == "__main__":
    task = "dolly"
    federated_learning(task)
