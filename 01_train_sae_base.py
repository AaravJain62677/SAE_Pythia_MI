import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#Config 

D_MODEL = 768
EXPANSION = 8          
L1_COEFF = 8          

cfg = LanguageModelSAERunnerConfig(

    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_MODEL * EXPANSION,         
        l1_coefficient=L1_COEFF,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),

    # Model
    model_name="EleutherAI/pythia-160m",
    hook_name="blocks.8.hook_resid_post",   

    # Dataset 
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,


    training_tokens=5_000_000,             
    train_batch_size_tokens=4096,
    context_size=128,                       
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    lr=5e-5,
    dtype="float32",
    seed=42,

    # Logging 
    logger=LoggingConfig(
        log_to_wandb=False,                 
    ),

    #  Output 
    checkpoint_path="checkpoints/sae_base",
    n_checkpoints=3,

    #  Hardware
    device=device,
)

#  Train
print("\nTraining SAE on BASE Pythia-160M")
print(f"  Model:    EleutherAI/pythia-160m")
print(f"  Hook:     blocks.8.hook_resid_post")
print(f"  Features: {D_MODEL * EXPANSION} ({EXPANSION}x expansion)")
print(f"  Tokens:   50M")
print(f"  Output:   checkpoints/sae_base/\n")

sae = LanguageModelSAETrainingRunner(cfg).run()

print("\nSAE saved to checkpoints/sae_base/")
print("Next: run 02_finetune.py")