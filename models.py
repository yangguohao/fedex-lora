from peft import (
    get_peft_model,
    TaskType,
    LoraConfig,
)
from transformers import RobertaForSequenceClassification, GPT2LMHeadModel, LlamaForCausalLM, AutoTokenizer


def create_llama3_base_model(args, use_unsloth=True):
    from unsloth import FastLanguageModel
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="meta-llama/Llama-3.2-1B",
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,  # Use 4bit quantization to reduce memory usage. Can be False
            dtype=args.dtype
        )
    else:
        if args.load_in_4bit:
            # TODO maybe use HQQ quant_method for 2 or 3-bit quantization
            quantization_config = {
                    "bnb_4bit_compute_dtype": args.dtype,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "load_in_4bit": True,
                    "quant_method": "bitsandbytes",
            }
        else:
            quantization_config = None
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",
                                                 torch_dtype=args.dtype,
                                                 quantization_config=quantization_config)

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    return model, tokenizer


def create_peft_llama_model_dolly(args, model, lora_dict=None, use_unsloth=True):
    from unsloth import FastLanguageModel
    if use_unsloth:
        if lora_dict is None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,  # Dropout = 0 is currently optimized
                bias="none",  # Bias = "none" is currently optimized
                use_rslora=args.rslora,
                use_gradient_checkpointing=args.gradient_checkpointing,
                random_state=args.seed,
                max_seq_length=args.max_seq_length,
            )
        else:
            for rank, modules in lora_dict.items():
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=rank,
                    target_modules=modules,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,  # Dropout = 0 is currently optimized
                    bias="none",  # Bias = "none" is currently optimized
                    use_rslora=args.rslora,
                    use_gradient_checkpointing=args.gradient_checkpointing,
                    random_state=args.seed,
                    max_seq_length=args.max_seq_length,
                )
    else:
        if lora_dict is None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_rslora=args.rslora,
                target_modules=["q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",],
            )

            model = get_peft_model(model, peft_config)
        else:
            raise Exception("Haven't implemented yet")
    return model


def create_peft_llama_model_dolly_ffa(args, model, lora_dict=None):
    from unsloth import FastLanguageModel
    if lora_dict is None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,  # Dropout = 0 is currently optimized
            bias="none",  # Bias = "none" is currently optimized
            use_rslora=args.rslora,
            use_gradient_checkpointing=args.gradient_checkpointing,
            random_state=args.seed,
            max_seq_length=args.max_seq_length,
        )
    else:
        for rank, modules in lora_dict.items():
            model = FastLanguageModel.get_peft_model(
                model,
                r=rank,
                target_modules=modules,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,  # Dropout = 0 is currently optimized
                bias="none",  # Bias = "none" is currently optimized
                use_rslora=args.rslora,
                use_gradient_checkpointing=args.gradient_checkpointing,
                random_state=args.seed,
                max_seq_length=args.max_seq_length,
            )
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False
    return model


def create_peft_model(model, args):

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value"],
    )

    model = get_peft_model(model, peft_config)

    return model


def create_peft_FFA_model(num_labels, args):

    model = RobertaForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.rslora,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, peft_config)

    # Make LoRA A matrices non-trainable
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    return model


def create_peft_gpt2_model_e2e(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model


def create_peft_gpt2_model_e2e_ffa(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define LoRA configuration for language modeling task
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # For language modeling
        inference_mode=False,
        r=args.lora_r,  # The dimension of the low-rank update matrices
        lora_alpha=args.lora_alpha,  # The scaling factor for LoRA layers
        lora_dropout=args.lora_dropout,  # Dropout to apply to LoRA layers
        target_modules=["c_attn", "c_proj"],  # Modules to apply LoRA
    )

    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    # Apply LoRA to the GPT-2 model
    model = get_peft_model(model, lora_config)
    return model
