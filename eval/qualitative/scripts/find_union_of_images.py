import pandas as pd

df = pd.read_csv("db/clean_up/dataset_no_duplicate_rows.csv")

KEY_COLS = ["city", "city_from_basename", "seq_from_basename", "frame_from_basename"]
MODEL_COL = "model"

# Unique image keys
num_keys = df[KEY_COLS].drop_duplicates().shape[0]
print("Total unique image keys:", num_keys)

# All models
all_models = sorted(df[MODEL_COL].unique())
print("Models:", all_models, " (", len(all_models), ")")

# For each key: how many models does it have?
per_key = (
    df.groupby(KEY_COLS)[MODEL_COL]
      .nunique()
      .reset_index(name="n_models_for_key")
)
print(len(df[MODEL_COL].unique().tolist()))
print("\nDistribution of #models per key:")
print(per_key["n_models_for_key"].value_counts().sort_index())

print("\nMax models any key has:")
print(per_key["n_models_for_key"].max())
