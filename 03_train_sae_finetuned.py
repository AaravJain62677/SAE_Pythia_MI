"""
03_train_sae_finetuned.py
─────────────────────────
Phase 3 (part 1): Train a second SAE on the FINE-TUNED Pythia-160M.

Critical: identical config to 01_train_sae_base.py — same architecture,
same hyperparameters, same training corpus (OpenWebText, NOT Python code).

Why same corpus? We want differences between SAE-1 and SAE-2 to reflect
changes in the MODEL's representations, not changes in the input distribution.
If we used Python code here, the SAE would just learn code features because
that's all it saw — not because the model changed.

Run:
    python scripts/03_train_sae_finetuned.py
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

# ── Config (IDENTICAL to 01 except model_name + checkpoint_path) ─────────────
D_MODEL = 768
EXPANSION = 8
L1_COEFF = 8

FINETUNED_MODEL_PATH = "checkpoints/pythia_finetuned"   # from Phase 2

cfg = LanguageModelSAERunnerConfig(

    # ── SAE architecture (same as base) ─────────────────────────────────────
    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_MODEL * EXPANSION,
        l1_coefficient=L1_COEFF,
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",
    ),

    # ── Fine-tuned model ─────────────────────────────────────────────────────
    # SAELens can load from a local HuggingFace-format directory
    model_name=FINETUNED_MODEL_PATH,
    hook_name="blocks.8.hook_resid_post",   # same layer as Phase 1

    # ── Dataset (SAME as Phase 1 — critical for fair comparison) ────────────
    dataset_path="Skylion007/openwebtext",
    is_dataset_tokenized=False,
    streaming=True,

    # ── Training (SAME as Phase 1) ───────────────────────────────────────────
    training_tokens=50_000_000,
    train_batch_size_tokens=4096,
    context_size=128,
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    lr=5e-5,
    dtype="float32",
    seed=42,

    # ── Logging ──────────────────────────────────────────────────────────────
    logger=LoggingConfig(log_to_wandb=False),

    # ── Output ───────────────────────────────────────────────────────────────
    checkpoint_path="checkpoints/sae_finetuned",
    n_checkpoints=3,

    device=device,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\n[Phase 3] Training SAE on FINE-TUNED Pythia-160M...")
print(f"  Model:    {FINETUNED_MODEL_PATH}")
print(f"  Hook:     blocks.8.hook_resid_post")
print(f"  Features: {D_MODEL * EXPANSION}")
print(f"  Dataset:  OpenWebText (same as Phase 1 — fair comparison)\n")

sae = LanguageModelSAETrainingRunner(cfg).run()

print("\n[Phase 3] Done. SAE saved to checkpoints/sae_finetuned/")
print("Next: run 04_compare_features.py")