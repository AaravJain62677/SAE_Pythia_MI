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
MAX_STEPS = 1000          # ~1-2 epochs on a 50M token subset is enough to shift representations
CONTEXT_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8            # effective batch = 32(8*4)

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
model.gradient_checkpointing_enable()   # saves ~30% VRAM

print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Dataset: Python code from The Stack
print("\nLoad Python code dataset (streaming)")

# The Stack is huge; we stream and take a subset
raw_dataset = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

def tokenize(example):
    return tokenizer(
        example["content"],         # The Stack field for source code
        truncation=True,
        max_length=CONTEXT_LENGTH,
        padding="max_length",
        return_tensors=None,
    )

# Take first 100K examples from stream, tokenize
print("Tokenizing 100K Python file")
dataset = raw_dataset.take(100_000)
dataset = dataset.map(tokenize, batched=True, batch_size=1000, remove_columns=["content", "size", "ext", "lang", "max_stars_repo_path", "max_stars_repo_name", "max_stars_count", "max_issues_repo_path", "max_issues_repo_name", "max_issues_count", "max_forks_repo_path", "max_forks_repo_name", "max_forks_count"])

# Convert streaming dataset to regular for Trainer
print("Collecting dataset into memory (this takes ~5 min)")
dataset = dataset.take(50_000)          # 50K files × 512 tokens = 25.6M tokens
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
    fp16=False,                 # float32 for stability on 3050
    dataloader_num_workers=2,
    report_to="none",           # disable wandb/tensorboard
    seed=42,
)

# Trainer
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                  # causal LM, not masked
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
print(f"  Dataset:     bigcode/the-stack (Python)")
print(f"  Max steps:   {MAX_STEPS}")
print(f"  Output:      {OUTPUT_DIR}\n")

trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n Fine-tuned model saved to {OUTPUT_DIR}/")
print("Next: run 03_train_sae_finetuned.py")