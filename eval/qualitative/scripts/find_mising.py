import pandas as pd

PANEL_100 = "db/clean_up/panel_100_keys.csv"
DEDUP = "db/clean_up/dataset_no_duplicate_rows.csv"
MISSING_PER_MODEL_CSV = "db/clean_up/missing_per_model_for_panel_100.csv"

KEY_COLS = ["city", "city_from_basename", "seq_from_basename", "frame_from_basename"]
MODEL_COL = "model"

df = pd.read_csv(DEDUP)
panel_100 = pd.read_csv(PANEL_100)

# Restrict dataset to just those 100 keys
df_100 = df.merge(panel_100, on=KEY_COLS, how="inner")

# All models we care about
all_models = sorted(df_100[MODEL_COL].unique())
print("Models in this subset:", all_models)

# Build the full grid: 100 keys × all models
keys_df = panel_100.copy()
keys_df["_tmp"] = 1

models_df = pd.DataFrame({MODEL_COL: all_models})
models_df["_tmp"] = 1

full = keys_df.merge(models_df, on="_tmp").drop(columns="_tmp")

# Count how many rows we have for each (key, model)
gm = (
    df_100.groupby(KEY_COLS + [MODEL_COL])
          .size()
          .reset_index(name="count")
)

full = full.merge(gm, on=KEY_COLS + [MODEL_COL], how="left")

# Missing = no row for that (key, model)
missing = full[full["count"].isna()].copy()

# Save for inspection
missing.to_csv(MISSING_PER_MODEL_CSV, index=False)

print("Total (key, model) combos in panel:", len(full))
print("Missing combos (need to run these):", len(missing))
print("Saved missing list to:", MISSING_PER_MODEL_CSV)

# Summary per model
print("\nMissing counts per model in the 100 panel:")
print(missing[MODEL_COL].value_counts())
