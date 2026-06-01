import torch
from transformer_lens import HookedTransformer
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

D_MODEL = 768
EXPANSION = 8
L1_COEFF = 8
FINETUNED_MODEL_PATH = "checkpoints/pythia_finetuned"
print("Loading fine-tuned model into TransformerLens...")
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH)

model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-160m",
    hf_model=hf_model,
    device=device,)
cfg = LanguageModelSAERunnerConfig(
    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_MODEL * EXPANSION,
        l1_coefficient=L1_COEFF,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),
    model_name="EleutherAI/pythia-160m",
    hook_name="blocks.8.hook_resid_post",
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
    logger=LoggingConfig(log_to_wandb=False),
    checkpoint_path="checkpoints/sae_finetuned",
    n_checkpoints=3,
    device=device,
)

print("\n Training SAE on FINE-TUNED Pythia-160M")
sae = LanguageModelSAETrainingRunner(cfg, model=model).run()
print("\nSAE Saved to Checkpoint")