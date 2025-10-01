import pandas as pd
import numpy as np

df = pd.read_csv("db/complete_dataset.csv", on_bad_lines="skip")

rubric_cols = [
    "visibility_restoration",
    "boundary_clarity",
    "scene_consistency",
    "object_consistency",
    "perceived_detectability",
    "relation_consistency",
]

for c in rubric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Per-image total
df["total_score"] = df[rubric_cols].sum(axis=1)

# Per-model counts
counts = df.groupby("model")["basename"].nunique().reset_index(name="n_images")
print("\nPer-model counts:\n", counts)

# Per-model total stats
stats_total = (
    df.groupby("model")["total_score"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .rename(columns={"mean":"mean_total", "std":"std_total", "count":"n"})
)
stats_total["sem_total"] = stats_total["std_total"] / np.sqrt(stats_total["n"])
print("\nPer-model total_score stats (mean/std/SEM):\n", stats_total)

# Optional: per-rubric stats
rubric_means = df.groupby("model")[rubric_cols].mean().add_suffix("_mean")
rubric_stds  = df.groupby("model")[rubric_cols].std(ddof=1).add_suffix("_std")
rubric_counts = df.groupby("model")[rubric_cols].count().iloc[:, :1]
rubric_counts.columns = ["n"]
rubric_stats = rubric_means.join(rubric_stds).join(rubric_counts)
for c in rubric_cols:
    rubric_stats[f"{c}_sem"] = rubric_stats[f"{c}_std"] / np.sqrt(rubric_stats["n"])
rubric_stats = rubric_stats.reset_index()
print("\nPer-model rubric stats (means/std/SEM):\n", rubric_stats)

# Save, if you want
df.to_csv("results/original_with_totals.csv", index=False)
stats_total.to_csv("results/original_per_model_total_stats.csv", index=False)
rubric_stats.to_csv("results/original_per_model_rubric_stats.csv", index=False)
