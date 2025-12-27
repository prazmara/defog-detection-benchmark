import pandas as pd

INPUT = "db/dataset_with_city_seq.csv"
OUTPUT_NO_DUPE = "db/clean_up/dataset_no_duplicate_rows.csv"

KEY_COLS = ["city", "city_from_basename", "seq_from_basename", "frame_from_basename"]
MODEL_COL = "model"

df = pd.read_csv(INPUT)

# Remove *exact* duplicate key+model rows, keep the first occurrence
df_no_dupe = df.drop_duplicates(subset=KEY_COLS + [MODEL_COL], keep="first")

print("Original rows:", len(df))
print("After dedup rows:", len(df_no_dupe))

df_no_dupe.to_csv(OUTPUT_NO_DUPE, index=False)
