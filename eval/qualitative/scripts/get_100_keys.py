import pandas as pd

df = pd.read_csv("db/clean_up/dataset_no_duplicate_rows.csv")
PANEL_100 = "db/clean_up/panel_100_keys.csv"
KEY_COLS = ["city", "city_from_basename", "seq_from_basename", "frame_from_basename"]
MODEL_COL = "model"


# Count how many models each key has
per_key = (
    df.groupby(KEY_COLS)[MODEL_COL]
      .nunique()
      .reset_index(name="n_models_for_key")
)

# Sort by number of models (desc), then arbitrary stable order
per_key_sorted = per_key.sort_values("n_models_for_key", ascending=False)

# Pick top 100 keys
panel_100 = per_key_sorted.head(100)[KEY_COLS].copy()
panel_100.to_csv(PANEL_100, index=False)

print("Total keys in dataset:", len(per_key))
print("Chosen keys for 100-panel:", len(panel_100))
print("Saved to:", PANEL_100)