import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

#  Paths
BASE_MODEL   = "EleutherAI/pythia-160m"
FT_MODEL     = "checkpoints/pythia_finetuned"
RESULTS_DIR  = Path("results")
QUAL_DIR     = RESULTS_DIR / "qualitative"
QUAL_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

#  Load everything
print("Loading models and SAEs")

def find_latest_checkpoint(base_path):
    path = Path(base_path)
    checkpoints = sorted(path.glob("final_*"), key=lambda x: int(x.name.split("_")[1]) if "_" in x.name else 0)
    return str(checkpoints[-1]) if checkpoints else base_path

sae_base = SAE.load_from_pretrained(find_latest_checkpoint("checkpoints/sae_base"), device=device)
sae_ft   = SAE.load_from_pretrained(find_latest_checkpoint("checkpoints/sae_finetuned"), device=device)

model_base = HookedTransformer.from_pretrained(BASE_MODEL, device=device)
model_ft   = HookedTransformer.from_pretrained(FT_MODEL,   device=device)

df = pd.read_csv(RESULTS_DIR / "feature_comparison.csv")
with open(RESULTS_DIR / "summary_stats.json") as f:
    summary = json.load(f)

# Pick features to analyze
# Features where direction drifted most (low cosine sim)
drifted_features = df[df["category"] == "drifted"].nsmallest(15, "cosine_sim")["feature_id_base"].tolist()
# Features with biggest frequency increase (likely code-related — expected)
increased_features = df.nlargest(10, "freq_shift")["feature_id_base"].tolist()
# Features with biggest frequency decrease (unexpected — most interesting for report)
decreased_features = df.nsmallest(10, "freq_shift")["feature_id_base"].tolist()

features_to_analyze = list(set(drifted_features + increased_features[:5] + decreased_features[:5]))
print(f"Analyzing {len(features_to_analyze)} features of interest")

#  Sample corpus for finding top activations 
print("Loading sample texts")
raw = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
sample_texts = [x["text"] for x in raw.take(2000)]

# Also add some Python code samples to see if code features activate
code_raw = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True, trust_remote_code=True)
code_samples = [x["content"][:300] for x in code_raw.take(200)]

all_texts = sample_texts + code_samples

# Function: find top-activating windows for a feature 
def get_top_activating_sequences(model, sae, feature_idx, texts, top_k=10, window=10):
    """
    Find the top-k token windows where a specific SAE feature activates most strongly.
    Returns list of (activation_value, token_string) tuples.
    """
    hook_name = "blocks.8.hook_resid_post"
    all_activations = []   

    model.eval()
    with torch.no_grad():
        for text in texts:
            tokens = model.to_tokens(text[:400], prepend_bos=True)
            if tokens.shape[1] < 5:
                continue
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            acts = cache[hook_name][0]          
            features = sae.encode(acts.to(device))  
            feat_acts = features[:, feature_idx].cpu()  

            # Find positions where this feature fires
            for pos in range(len(feat_acts)):
                val = feat_acts[pos].item()
                if val > 0:
                    # Extract window around this position
                    start = max(0, pos - window // 2)
                    end   = min(tokens.shape[1], pos + window // 2)
                    window_tokens = tokens[0, start:end]
                    context = model.to_string(window_tokens)
                    all_activations.append((val, context))

    # Sort by activation value, return top-k
    all_activations.sort(key=lambda x: x[0], reverse=True)
    return all_activations[:top_k]

#  Analyze each feature 
print("\nFinding top-activating sequences")

results = {}

for feat_id in tqdm(features_to_analyze, desc="Analyzing features"):
    feat_id = int(feat_id)
    matched_ft_id = int(df[df["feature_id_base"] == feat_id]["feature_id_ft"].values[0])
    cosine_sim    = float(df[df["feature_id_base"] == feat_id]["cosine_sim"].values[0])
    freq_shift    = float(df[df["feature_id_base"] == feat_id]["freq_shift"].values[0])
    category      = df[df["feature_id_base"] == feat_id]["category"].values[0]

    # Top activations from BASE model/SAE
    base_top = get_top_activating_sequences(model_base, sae_base, feat_id, all_texts)
    # Top activations from FT model/SAE (using matched feature index)
    ft_top   = get_top_activating_sequences(model_ft,   sae_ft,   matched_ft_id, all_texts)

    results[feat_id] = {
        "feature_id_base": feat_id,
        "feature_id_ft":   matched_ft_id,
        "cosine_sim":      cosine_sim,
        "freq_shift":      freq_shift,
        "category":        category,
        "base_top_activations": base_top,
        "ft_top_activations":   ft_top,
    }

    # Save to text file
    out_path = QUAL_DIR / f"feature_{feat_id:04d}.txt"
    with open(out_path, "w") as f:
        f.write(f"Feature {feat_id} (Base) → Feature {matched_ft_id} (FT)\n")
        f.write(f"Cosine similarity: {cosine_sim:.4f}  |  Category: {category}  |  Freq shift: {freq_shift:+.4f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("TOP ACTIVATING SEQUENCES — BASE MODEL:\n")
        f.write("-" * 40 + "\n")
        for val, ctx in base_top:
            f.write(f"[{val:.3f}] {repr(ctx)}\n")

        f.write("\n\nTOP ACTIVATING SEQUENCES — FINE-TUNED MODEL:\n")
        f.write("-" * 40 + "\n")
        for val, ctx in ft_top:
            f.write(f"[{val:.3f}] {repr(ctx)}\n")

        # Quick human interpretation prompt
        f.write("\n\nMANUAL LABEL\n")
        f.write("  Base feature meaning:\n")
        f.write("  FT feature meaning:\n")
        f.write("  Unexpected drift?      YES / NO\n")
        f.write("  Notes: \n")

# Generate report-ready summary 
print("\nGenerating qualitative summary for report")

summary_path = QUAL_DIR / "QUALITATIVE_SUMMARY.txt"
with open(summary_path, "w") as f:
    f.write("QUALITATIVE FEATURE ANALYSIS SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("MOST DRIFTED FEATURES (low cosine similarity — direction changed):\n")
    f.write("-" * 60 + "\n")
    for feat_id in drifted_features[:8]:
        feat_id = int(feat_id)
        r = results.get(feat_id)
        if r:
            f.write(f"\nFeature {feat_id:4d} | sim={r['cosine_sim']:.3f} | freq_shift={r['freq_shift']:+.4f}\n")
            f.write(f"  BASE top-3: {[ctx for _, ctx in r['base_top_activations'][:3]]}\n")
            f.write(f"  FT   top-3: {[ctx for _, ctx in r['ft_top_activations'][:3]]}\n")

    f.write("\n\nMOST INCREASED ACTIVATION (likely code features — expected drift):\n")
    f.write("-" * 60 + "\n")
    for feat_id in increased_features[:5]:
        feat_id = int(feat_id)
        r = results.get(feat_id)
        if r:
            f.write(f"\nFeature {feat_id:4d} | sim={r['cosine_sim']:.3f} | freq_shift={r['freq_shift']:+.4f}\n")
            f.write(f"  BASE top-3: {[ctx for _, ctx in r['base_top_activations'][:3]]}\n")
            f.write(f"  FT   top-3: {[ctx for _, ctx in r['ft_top_activations'][:3]]}\n")

    f.write("\n\nMOST DECREASED ACTIVATION (unexpected drift candidates):\n")
    f.write("-" * 60 + "\n")
    for feat_id in decreased_features[:5]:
        feat_id = int(feat_id)
        r = results.get(feat_id)
        if r:
            f.write(f"\nFeature {feat_id:4d} | sim={r['cosine_sim']:.3f} | freq_shift={r['freq_shift']:+.4f}\n")
            f.write(f"  BASE top-3: {[ctx for _, ctx in r['base_top_activations'][:3]]}\n")
            f.write(f"  FT   top-3: {[ctx for _, ctx in r['ft_top_activations'][:3]]}\n")

print(f"\nSaved qualitative analysis to results/qualitative/")