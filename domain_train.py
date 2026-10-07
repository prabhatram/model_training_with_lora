from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model
)


MODEL_NAME = "Qwen/Qwen3-0.6B"


# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Dataset

dataset = load_dataset(
    "json",
    data_files=
        "data/domain_corpus_clean.jsonl",
    split="train"
)


dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)


def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256
    )


tokenized = dataset.map(
    tokenize,
    remove_columns=
        dataset["train"].column_names
)



# Model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto"
)


# LoRA

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()


# Data collator

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# Training configuration

training_args = TrainingArguments(
    output_dir=
        "models/domain_adapter",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)


# Trainer

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=
        tokenized["train"],

    eval_dataset=
        tokenized["test"],

    data_collator=collator
)


# Train

trainer.train()


# Save adapter

model.save_pretrained(
    "models/domain_adapter"
)

tokenizer.save_pretrained(
    "models/domain_adapter"
)