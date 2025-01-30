import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from data_utils import *
from models import *
import argparse
import warnings
from sklearn.metrics import matthews_corrcoef
import wandb
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from opacus import PrivacyEngine
from opacus.validators.module_validator import ModuleValidator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
import torch
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from data_utils import *
import os
from copy import deepcopy


def train_client(model, dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.local_epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    scaler = GradScaler()
    model.train()
    for epoch in range(args.local_epochs):

        for step, data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}

            with autocast():
                outputs = model(**data)
                loss = outputs.loss

            wandb.log({"client_loss": loss.detach().cpu().numpy()})

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task == "mrpc":
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    elif task == "qqp":
        return accuracy_score(all_true_labels, all_predictions), f1_score(
            all_true_labels, all_predictions
        )
    elif task in ["mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_global_model(global_model, dataloader, args, max_metric1, max_metric2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    global_model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            outputs = global_model(**batch)

            eval_loss += outputs.loss.detach().cpu().numpy()

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(batch["labels"].cpu().numpy())

    eval_loss /= len(dataloader)

    # Calculate the metrics for the specific task
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    wandb.log(
        {
            f"eval_loss": eval_loss,
            f"metric1": metric1,
            f"metric2": metric2 if metric2 is not None else 0,
            f"max_metric1": max_metric1,
            f"max_metric2": max_metric2 if max_metric2 is not None else 0,
        }
    )

    return max_metric1, max_metric2


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_client_e2e(model, train_dataset, val_dataset, tokenizer, args):
    num_epochs = args.local_epochs  # or whatever number of epochs you want
    per_device_train_batch_size = args.batch_size
    num_training_steps = len(train_dataset) * num_epochs // per_device_train_batch_size
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps for warmup

    optimizer = torch.optim.AdamW(model.parameters())

    # Define training arguments
    training_args = TrainingArguments(
        # Directory to save the model
        output_dir="./models_trained/gpt4/dump/models/gpt2-e2e-lora_gpt4",
        overwrite_output_dir=True,
        logging_dir="./models_trained/gpt4/dump/logs/gpt2-e2e-lora_gpt4",  # Directory for logs
        per_device_train_batch_size=args.batch_size,  # Adjust based on your GPU capacity
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",
        num_train_epochs=num_epochs,  # Number of training epochs
        learning_rate=args.lr,  # Learning rate for LoRA parameters
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        report_to="wandb",
        run_name="fed-lora",
        logging_steps=100,  # Log every 100 steps
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(
            optimizer,
            get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps),
        ),
    )

    # Train the model
    trainer.train()
    return model.state_dict()


def gen_and_save(model, dataloader, tokenizer, args):
    device = args.device
    model.to(device)
    model.eval()

    all_predictions = []

    all_inputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}

            # Generate predictions (starting from after the MR)
            generated = model.generate(
                input_ids=inputs["input_ids"],  # Input MR as prompt
                attention_mask=inputs["attention_mask"],
                max_length=inputs["input_ids"].shape[1]
                           + 50,  # Allow space for generation after MR
                num_return_sequences=1,
                no_repeat_ngram_size=4,
                do_sample=True,
                num_beams=10,
                penalty_alpha=0.9,
                pad_token_id=tokenizer.eos_token_id,  # Ensure padding works correctly
            )
            # Decode the generated predictions, excluding the input MR tokens
            # We slice the generated tokens to remove the input MR part

            input_seq = tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
            predictions = [
                tokenizer.decode(
                    generated[i][len(inputs["input_ids"][i]):],
                    skip_special_tokens=True,
                )
                for i in range(generated.shape[0])
            ]
            # Collect predictions and references
            all_inputs.extend(input_seq)
            all_predictions.extend(predictions)
            # all_references.extend(references)

    return all_predictions, all_inputs


def process_lists(input_list, second_list, third_list):
    result1 = []
    result2 = []
    result3 = []
    current_group = []
    current_item = None
    second_list_index = 0

    for item in input_list:
        if item != current_item:
            if current_group:
                result1.append(current_group)
                result2.append(current_item)
                result3.append(third_list[second_list_index - 1])
            current_item = item
            current_group = [second_list[second_list_index]]
            second_list_index += 1
        else:
            if second_list_index < len(second_list):
                current_group.append(second_list[second_list_index])
                second_list_index += 1

    if current_group:
        result1.append(current_group)

    return result1, result2, result3


def evaluate_e2e_save_text(model, test_data, tokenizer, args):
    def preprocess_function2(examples):
        inputs = examples["meaning_representation"]
        targets = examples["human_reference"]

        # Combine the input-output pair into a single text
        model_inputs = [f"{input_} ->" for input_, target in zip(inputs, targets)]

        # Tokenize the combined inputs
        tokenized_inputs = tokenizer(
            model_inputs,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Labels are the same as input_ids but shift them for next-token prediction
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        # Set the labels to -100 where attention mask is 0 (this will ignore padding in loss computation)
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100

        return tokenized_inputs

    tokenized_test_dataset = test_data.map(preprocess_function2, batched=True)
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(
        ["meaning_representation", "human_reference"]
    )
    tokenized_test_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataloader = create_dataloader(tokenized_test_dataset, args)
    all_predictions, all_inputs = gen_and_save(model, test_dataloader, tokenizer, args)
    all_references = test_data[0: len(all_predictions)]["human_reference"]

    all_references_new, all_inputs_new, all_predictions_new = process_lists(
        all_inputs, all_references, all_predictions
    )

    path_pred = args.run_dir + "/predictions.txt"
    path_ref = args.run_dir + "/refs_exact.txt"

    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)

    with open(path_pred, "w") as file:
        for item in all_predictions_new:
            file.write(item.strip() + "\n")

    with open(path_ref, "w") as file:
        for str_list in all_references_new:
            for item in str_list:
                file.write(item.strip() + "\n")

            file.write("\n")


def evaluate_instruction_tuning_global(gt, ref):
    # TODO: add evaluation method for instruction tuning task rouge-l score
    class Scorer:
        def __init__(self):
            self.scorers = [
                (Bleu(), "Bleu"),
                (Rouge(), "ROUGE_L"),
            ]

        def compute(self, gt, ref):
            total_scores = {}
            for scorer, method in self.scorers:
                print('computing %s score...' % (scorer.method()))
                score, scores = scorer.compute_score(gt, ref)
                print(score)
                total_scores[method] = score[-1] if type(score) is list else score
            return total_scores

    gt = {i: [gt[i]] for i in range(len(gt))}
    ref = {i: [ref[i]] for i in range(len(ref))}
    total_scorer = Scorer()
    return total_scorer.compute(gt, ref)


def evaluate_instruction_tuning_save_text(model, test_data, tokenizer, args):
    def generate(model, dataloader, tokenizer, args):
        device = args.device
        model.to(device)
        model.eval()

        all_predictions = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                generated = model.generate(**inputs, max_new_tokens=128, use_cache=True)
                predictions = tokenizer.batch_decode(generated)
                predictions = [predictions[i].split("### Response: ")[-1] for i in range(len(predictions))]
                # print("Generated result:", predictions[0])
                # print("Response: ", dataloader['response'][step])
                # print("-"*64)
                all_predictions.extend(predictions)

        return all_predictions

    all_predictions = generate(model, DataLoader(test_data['text'], batch_size=64, shuffle=False), tokenizer, args)

    all_references = test_data['response']

    total_scores = evaluate_instruction_tuning_global(all_predictions, all_references)

    # path_pred = args.run_dir + "/predictions.txt"
    # path_ref = args.run_dir + "/refs_exact.txt"
    #
    # if not os.path.exists(args.run_dir):
    #     os.makedirs(args.run_dir)
    #
    # with open(path_pred, "w") as file:
    #     for item in all_predictions:
    #         file.write(item.strip() + "\n")
    #
    # with open(path_ref, "w") as file:
    #     for str_list in all_references:
    #         for item in str_list:
    #             file.write(item.strip() + "\n")
    #
    #         file.write("\n")
    return total_scores


def train_client_dolly(model, tokenizer, train, args):
    from trl import SFTConfig, SFTTrainer
    model.train()
    training_args = SFTConfig(learning_rate=args.lr,
                              lr_scheduler_type=args.lr_scheduler_type,
                              per_device_train_batch_size=args.batch_size,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              num_train_epochs=args.local_epochs,
                              fp16=True,
                              optim=args.optim,
                              weight_decay=args.weight_decay,
                              warmup_ratio=args.warmup_ratio,
                              output_dir=args.run_dir,
                              seed=args.seed,
                              # logging_steps=1,
                              save_strategy='no',
                              logging_strategy='no',
                              max_seq_length=args.max_seq_length,
                              packing=True)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        dataset_text_field="text",
        args=training_args,
    )

    trainer.train()
    return model
