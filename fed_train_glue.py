from fed_agg import *
from train_eval import *
from utils import *
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument(
    "--task", type=str, default="cola", help="GLUE task to fine-tune on"
)
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha value")
parser.add_argument(
    "--lora_dropout", type=float, default=0.1, help="LoRA dropout value"
)
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--agg_type", type=str, default="normal", help="Type of aggregation"
)
parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
parser.add_argument(
    "--local_epochs", type=int, default=3, help="Number of local epochs"
)
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument(
    "--max_seq_length", type=int, default=128, help="Maximum sequence length"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

wandb.init(project="test", config=args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def federated_learning(task):

    train_data, val_data, test_data = load_and_preprocess_data(task, args)

    num_labels = len(set(train_data["labels"].numpy()))

    if args.task == "stsb":
        num_labels = 1

    client_dataloaders = create_client_dataloaders(train_data, args)
    val_dataloader = create_dataloader(val_data, args)

    max_metric_1 = 0
    max_metric_2 = 0
    base_model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )
    if args.agg_type == "ffa":
        global_model = create_peft_FFA_model(num_labels, args)
    else:
        global_model = create_peft_model(base_model, args)

    k = "base_model.model.roberta.encoder.layer.0.attention.self.query.lora_A.default.weight"
    for r in range(args.rounds):
        print(f"Round {r + 1}/{args.rounds}")
        client_models = []
        global_dict = global_model.state_dict()
        for i in range(args.num_clients):
            if args.agg_type == "ffa":
                client_model = create_peft_FFA_model(num_labels, args)
            else:
                client_model = create_peft_model(base_model, args)
            client_model.load_state_dict(global_dict)
            # print('Before Train', client_model.state_dict()[k])
            train_client(client_model, client_dataloaders[i], args)
            # print('After Train', client_model.state_dict()[k])
            client_models.append(client_model.state_dict())
        # print(client_models[0][k], client_models[1][k])
        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)
        elif args.agg_type == 'flora':
            global_model = aggregate_models_normal(global_model, client_models)

        max_metric_1, max_metric_2 = evaluate_global_model(
            global_model, val_dataloader, args, max_metric_1, max_metric_2
        )


# Main execution
if __name__ == "__main__":
    task = args.task
    federated_learning(task)
