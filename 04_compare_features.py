"""
04_compare_features.py
──────────────────────
Phase 3 (part 2): Compare features between SAE-base and SAE-finetuned.

This is the core scientific contribution of the project.

Pipeline:
  1. Load both SAEs
  2. Match features by decoder cosine similarity
  3. Compute quantitative drift metrics
  4. Find top-activating token sequences per feature (qualitative)
  5. Identify stable / drifted / new / dead features
  6. Save all results for the report

Output:
  results/feature_comparison.csv    — per-feature metrics
  results/summary_stats.json        — aggregate numbers for report table
  results/figures/                  — plots

Run:
    python scripts/04_compare_features.py
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from sae_lens import SAE

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_SAE_PATH      = "checkpoints/sae_base"
FT_SAE_PATH        = "checkpoints/sae_finetuned"
BASE_MODEL_NAME    = "EleutherAI/pythia-160m"
FT_MODEL_PATH      = "checkpoints/pythia_finetuned"
RESULTS_DIR        = Path("results")
FIGURES_DIR        = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Thresholds ───────────────────────────────────────────────────────────────
STABLE_THRESH  = 0.9    # cosine sim > 0.9  → feature is stable
DRIFTED_THRESH = 0.5    # 0.5 < sim < 0.9  → feature drifted
# sim < 0.5             → feature is new/dead (no good match)

# ── 1. Load SAEs ─────────────────────────────────────────────────────────────
print("Loading SAEs...")
# SAELens saves checkpoints as HuggingFace-format directories
# Find the final checkpoint (highest step number)
def find_latest_checkpoint(base_path):
    path = Path(base_path)
    checkpoints = sorted(path.glob("final_*"), key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0)
    if checkpoints:
        return str(checkpoints[-1])
    # fallback: use base path directly
    return base_path

base_ckpt = find_latest_checkpoint(BASE_SAE_PATH)
ft_ckpt   = find_latest_checkpoint(FT_SAE_PATH)

print(f"  Base SAE:       {base_ckpt}")
print(f"  Finetuned SAE:  {ft_ckpt}")

sae_base = SAE.load_from_pretrained(base_ckpt, device=device)
sae_ft   = SAE.load_from_pretrained(ft_ckpt,   device=device)

# Decoder weight matrices: shape (d_sae, d_model)
# Each row is the direction for one feature
W_dec_base = sae_base.W_dec.detach()   # (6144, 768)
W_dec_ft   = sae_ft.W_dec.detach()     # (6144, 768)

n_features = W_dec_base.shape[0]
print(f"  Features per SAE: {n_features}")

# ── 2. Feature matching by cosine similarity ─────────────────────────────────
print("\nComputing feature matching (cosine similarity)...")

# Normalize decoder directions to unit vectors
W_base_norm = W_dec_base / (W_dec_base.norm(dim=1, keepdim=True) + 1e-8)
W_ft_norm   = W_dec_ft   / (W_dec_ft.norm(dim=1, keepdim=True) + 1e-8)

# Cosine similarity matrix: (n_base_features, n_ft_features)
# Do in chunks to avoid OOM on 6GB VRAM
CHUNK = 512
sim_matrix = torch.zeros(n_features, n_features, device="cpu")

for i in tqdm(range(0, n_features, CHUNK), desc="Sim matrix"):
    chunk = W_base_norm[i:i+CHUNK].to(device)
    sim_matrix[i:i+CHUNK] = (chunk @ W_ft_norm.T).cpu()

# Greedy matching: for each base feature, find best-matching ft feature
best_match_idx  = sim_matrix.argmax(dim=1)          # (n_features,)
best_match_sim  = sim_matrix.max(dim=1).values      # (n_features,)

# ── 3. Categorize features ───────────────────────────────────────────────────
stable_mask  = best_match_sim >= STABLE_THRESH
drifted_mask = (best_match_sim >= DRIFTED_THRESH) & (~stable_mask)
dead_mask    = best_match_sim < DRIFTED_THRESH

n_stable  = stable_mask.sum().item()
n_drifted = drifted_mask.sum().item()
n_dead    = dead_mask.sum().item()

print(f"\nFeature drift summary:")
print(f"  Stable   (sim ≥ 0.9): {n_stable:4d} / {n_features}  ({100*n_stable/n_features:.1f}%)")
print(f"  Drifted  (0.5–0.9):   {n_drifted:4d} / {n_features}  ({100*n_drifted/n_features:.1f}%)")
print(f"  New/Dead (sim < 0.5): {n_dead:4d} / {n_features}  ({100*n_dead/n_features:.1f}%)")

# ── 4. Activation frequency analysis ─────────────────────────────────────────
# We need to run both SAEs on a shared corpus to get activation frequencies.
# Use a small sample of OpenWebText for speed.

print("\nComputing activation frequencies on 500 sentences...")

from transformer_lens import HookedTransformer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Load both models
model_base = HookedTransformer.from_pretrained("EleutherAI/pythia-160m", device=device)
model_ft   = HookedTransformer.from_pretrained(FT_MODEL_PATH, device=device)

# Sample text
raw = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
sample_texts = [x["text"] for x in raw.take(500)]

def get_activation_frequencies(model, sae, texts, hook_name="blocks.8.hook_resid_post"):
    """Count how often each SAE feature fires across a text sample."""
    freq = torch.zeros(sae.cfg.d_sae, device="cpu")
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="  Computing freqs", leave=False):
            tokens = model.to_tokens(text[:512], prepend_bos=True)  # truncate long texts
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            acts = cache[hook_name][0]      # (seq_len, d_model)
            features = sae.encode(acts.to(device))   # (seq_len, d_sae)
            freq += (features > 0).float().sum(dim=0).cpu()
            total_tokens += acts.shape[0]

    return freq / total_tokens    # frequency = fraction of tokens where feature fires

print("  Base model frequencies...")
freq_base = get_activation_frequencies(model_base, sae_base, sample_texts)
print("  Fine-tuned model frequencies...")
freq_ft   = get_activation_frequencies(model_ft,   sae_ft,   sample_texts)

# Frequency shift per feature (matched pairs)
freq_ft_matched = freq_ft[best_match_idx]
freq_shift = freq_ft_matched - freq_base     # positive = fires more after FT

# ── 5. Build results dataframe ────────────────────────────────────────────────
print("\nBuilding results dataframe...")

df = pd.DataFrame({
    "feature_id_base":      range(n_features),
    "feature_id_ft":        best_match_idx.numpy(),
    "cosine_sim":           best_match_sim.numpy(),
    "freq_base":            freq_base.numpy(),
    "freq_ft":              freq_ft_matched.numpy(),
    "freq_shift":           freq_shift.numpy(),
    "category": pd.Categorical(
        np.where(stable_mask.numpy(),  "stable",
        np.where(drifted_mask.numpy(), "drifted", "new_dead"))
    ),
})

df.to_csv(RESULTS_DIR / "feature_comparison.csv", index=False)
print(f"  Saved: results/feature_comparison.csv")

# ── 6. Summary stats for report ───────────────────────────────────────────────
summary = {
    "n_features": n_features,
    "n_stable":   n_stable,
    "n_drifted":  n_drifted,
    "n_dead":     n_dead,
    "pct_stable":  round(100 * n_stable  / n_features, 1),
    "pct_drifted": round(100 * n_drifted / n_features, 1),
    "pct_dead":    round(100 * n_dead    / n_features, 1),
    "mean_cosine_sim":  round(best_match_sim.mean().item(), 4),
    "median_cosine_sim": round(best_match_sim.median().item(), 4),
    "mean_freq_shift":  round(freq_shift.mean().item(), 6),
    # Top 10 most frequency-increased features (likely code-related)
    "top_increased_features": df.nlargest(10, "freq_shift")["feature_id_base"].tolist(),
    # Top 10 most frequency-decreased features (potentially unexpected drift)
    "top_decreased_features": df.nsmallest(10, "freq_shift")["feature_id_base"].tolist(),
}

with open(RESULTS_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: results/summary_stats.json")

# ── 7. Figures ────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Fig 1: Cosine similarity distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("SAE Feature Drift: Base → Fine-tuned Pythia-160M (Layer 8)", fontsize=13)

axes[0].hist(best_match_sim.numpy(), bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3)
axes[0].axvline(STABLE_THRESH,  color="green",  linestyle="--", label=f"Stable (>{STABLE_THRESH})")
axes[0].axvline(DRIFTED_THRESH, color="orange", linestyle="--", label=f"Drifted (>{DRIFTED_THRESH})")
axes[0].set_xlabel("Cosine Similarity (Base vs FT Feature Direction)")
axes[0].set_ylabel("Count")
axes[0].set_title("Feature Matching Similarity Distribution")
axes[0].legend(fontsize=8)

# Fig 2: Category breakdown pie
labels = ["Stable", "Drifted", "New/Dead"]
sizes  = [n_stable, n_drifted, n_dead]
colors = ["#2ca02c", "#ff7f0e", "#d62728"]
axes[1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Feature Category Breakdown")

# Fig 3: Activation frequency shift distribution
axes[2].hist(freq_shift.numpy(), bins=100, color="#DD8452", edgecolor="white", linewidth=0.3)
axes[2].axvline(0, color="black", linestyle="-", linewidth=1.5)
axes[2].set_xlabel("Activation Frequency Shift (FT - Base)")
axes[2].set_ylabel("Count")
axes[2].set_title("How Much Feature Activation Frequencies Changed")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_feature_drift_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: results/figures/fig1_feature_drift_overview.png")

# Fig 4: Scatter — cosine sim vs frequency shift
plt.figure(figsize=(8, 5))
scatter_colors = df["category"].map({"stable": "#2ca02c", "drifted": "#ff7f0e", "new_dead": "#d62728"})
plt.scatter(df["cosine_sim"], df["freq_shift"], c=scatter_colors, alpha=0.3, s=5)
plt.xlabel("Cosine Similarity (feature direction preservation)")
plt.ylabel("Activation Frequency Shift")
plt.title("Feature Direction Preservation vs. Activation Change")
from matplotlib.patches import Patch
legend_elements = [Patch(fc="#2ca02c", label="Stable"), Patch(fc="#ff7f0e", label="Drifted"), Patch(fc="#d62728", label="New/Dead")]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_sim_vs_freq_shift.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: results/figures/fig2_sim_vs_freq_shift.png")

print("\n[Phase 3 Part 2] Feature comparison complete.")
print("Next: run 05_qualitative_analysis.py to find top-activating sequences")
print("\nKey numbers for your report:")
print(json.dumps(summary, indent=2))