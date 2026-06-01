"""
01_train_sae_base.py
────────────────────
Phase 1: Train a Sparse Autoencoder on layer 8 of the BASE Pythia-160M.

Uses SAELens v6 API with StandardTrainingSAEConfig.
Trains on OpenWebText (general corpus) so features reflect general language.

Run:
    python scripts/01_train_sae_base.py
"""

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

# ── Device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Config ───────────────────────────────────────────────────────────────────
# Pythia-160M: d_model = 768, 12 layers
# We hook at layer 8 (resid_post) — upper-middle, rich semantic features

D_MODEL = 768
EXPANSION = 8          # 768 * 8 = 6144 SAE features
L1_COEFF = 8           # sparsity penalty; target L0 of 20-50

cfg = LanguageModelSAERunnerConfig(

    # ── SAE architecture ────────────────────────────────────────────────────
    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_MODEL * EXPANSION,          # 6144 features
        l1_coefficient=L1_COEFF,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),

    # ── Model ──────────────────────────────────────────────────
    model_name="EleutherAI/pythia-160m",
    hook_name="blocks.8.hook_resid_post",   # residual stream after layer 8

    # ── Dataset ─────────────────────────────────────────────────────────────
    # General corpus so features represent general language
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,

    # ── Training ────────────────────────────────────────────────────────────
    training_tokens=50_000_000,             # 50M tokens
    train_batch_size_tokens=4096,
    context_size=128,                       # sequence chunk length
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    lr=5e-5,
    dtype="float32",
    seed=42,

    # ── Logging ─────────────────────────────────────────────────────────────
    logger=LoggingConfig(
        log_to_wandb=False,                 # set True if you have wandb
    ),

    # ── Output ──────────────────────────────────────────────────────────────
    checkpoint_path="checkpoints/sae_base",
    n_checkpoints=3,

    # ── Hardware ────────────────────────────────────────────────────────────
    device=device,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\n[Phase 1] Training SAE on BASE Pythia-160M...")
print(f"  Model:    EleutherAI/pythia-160m")
print(f"  Hook:     blocks.8.hook_resid_post")
print(f"  Features: {D_MODEL * EXPANSION} ({EXPANSION}x expansion)")
print(f"  Tokens:   50M")
print(f"  Output:   checkpoints/sae_base/\n")

sae = LanguageModelSAETrainingRunner(cfg).run()

print("\n[Phase 1] Done. SAE saved to checkpoints/sae_base/")
print("Next: run 02_finetune.py")