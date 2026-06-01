import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

#  Device 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Config 
D_MODEL = 768
EXPANSION = 8
L1_COEFF = 8

FINETUNED_MODEL_PATH = "checkpoints/pythia_finetuned"   

cfg = LanguageModelSAERunnerConfig(

    #SAE architecture (same as base)
    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_MODEL * EXPANSION,
        l1_coefficient=L1_COEFF,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),

    #  Fine-tuned model
    # SAELens can load from a local HuggingFace-format directory
    model_name=FINETUNED_MODEL_PATH,
    hook_name="blocks.8.hook_resid_post",   

    # Dataset
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,

    # Training
    training_tokens=5_000_000,
    train_batch_size_tokens=4096,
    context_size=128,
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    lr=5e-5,
    dtype="float32",
    seed=42,

    #Logging            
    logger=LoggingConfig(log_to_wandb=False),

    # Output
    checkpoint_path="checkpoints/sae_finetuned",
    n_checkpoints=3,

    device=device,
)

# Train
print("\n Training SAE on FINE-TUNED Pythia-160M")
print(f"  Model:    {FINETUNED_MODEL_PATH}")
print(f"  Hook:     blocks.8.hook_resid_post")
print(f"  Features: {D_MODEL * EXPANSION}")
print(f"  Dataset:  OpenWebText \n")

sae = LanguageModelSAETrainingRunner(cfg).run()

print("\n SAE saved to checkpoints/sae_finetuned/")
print("Next: run 04_compare_features.py")