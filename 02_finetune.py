import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Config 
MODEL_NAME = "EleutherAI/pythia-160m"
OUTPUT_DIR = "checkpoints/pythia_finetuned"
MAX_STEPS = 1000          
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8           

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load tokenizer + model
print(f"\nLoading {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Pythia has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)
model.gradient_checkpointing_enable()   

print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

print("\nLoad Python code dataset (streaming)")

raw_dataset = load_dataset(
    "flytech/python-codes-25k",
    split="train",
    streaming=True,
)
def tokenize(example):
    return tokenizer(
        example["output"],        
        truncation=True,
        max_length=CONTEXT_LENGTH,
        padding="max_length",
        return_tensors=None,
    )

print("Tokenizing 100K Python file")
dataset = raw_dataset.take(100_000)
dataset = dataset.map(tokenize, batched=True, batch_size=1000, remove_columns=dataset.column_names)


print("Collecting dataset into memory")
dataset = dataset.take(50_000)         
dataset_list = list(dataset)

from datasets import Dataset
train_dataset = Dataset.from_list(dataset_list)
print(f"Dataset size: {len(train_dataset)} examples")

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=1e-5,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    fp16=False,                 
    dataloader_num_workers=0,
    report_to="none",          
    seed=42,
)

# Trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Train
print(f"\nFine-tuning on Python code")
print(f"  Base model:  {MODEL_NAME}")
print(f"  Dataset:     flytech/python-codes-25k")
print(f"  Max steps:   {MAX_STEPS}")
print(f"  Output:      {OUTPUT_DIR}\n")

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n Fine-tuned model saved to {OUTPUT_DIR}/")
print("Next: run 03_train_sae_finetuned.py")