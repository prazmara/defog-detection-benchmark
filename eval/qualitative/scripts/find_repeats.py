#!/usr/bin/env python3
import pandas as pd

INPUT_CSV = "db/dataset_with_city_seq.csv"

CLEAN_OUT = "db/test_stuff/dataset_clean_equal_models.csv"
MISSING_OUT = "db/test_stuff/image_model_missing.csv"
DUP_COMBOS_OUT = "db/test_stuff/image_model_duplicates_per_key_model.csv"
DUP_ROWS_OUT = "db/test_stuff/image_model_duplicate_rows.csv"

KEY_COLS = [
    "city",
    "city_from_basename",
    "seq_from_basename",
    "frame_from_basename",
]
MODEL_COL = "model"


def main():
    df = pd.read_csv(INPUT_CSV)

    # Sanity checks
    for col in KEY_COLS + [MODEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Global set of models we expect per image
    all_models = sorted(df[MODEL_COL].unique())
    num_models = len(all_models)
    print(f"Global models ({num_models}): {all_models}")

    # ------------------------------------------------------------------
    # 1) Count rows per (image key, model)
    # ------------------------------------------------------------------
    gm = (
        df.groupby(KEY_COLS + [MODEL_COL])
          .size()
          .reset_index(name="count")
    )

    # Combos with duplicates (count > 1)
    dup_combos = gm[gm["count"] > 1]

    # Get the actual duplicate *rows* from the original df
    dup_rows = df.merge(
        dup_combos[KEY_COLS + [MODEL_COL]],
        on=KEY_COLS + [MODEL_COL],
        how="inner",
    )

    # ------------------------------------------------------------------
    # 2) Find missing (image key, model) combos
    #    Build the full grid of all keys × all models, then see what's missing
    # ------------------------------------------------------------------
    keys_df = df[KEY_COLS].drop_duplicates().copy()
    keys_df["_tmp"] = 1

    models_df = pd.DataFrame({MODEL_COL: all_models})
    models_df["_tmp"] = 1

    # Cartesian product: all (key, model) pairs that *should* exist
    full = keys_df.merge(models_df, on="_tmp").drop(columns="_tmp")

    # Attach the count from gm (how many rows we actually have for each pair)
    full = full.merge(
        gm,
        on=KEY_COLS + [MODEL_COL],
        how="left",
    )

    # Missing means no row at all for that (key, model)
    missing = full[full["count"].isna()].copy()

    # ------------------------------------------------------------------
    # 3) Summary per model (missing & duplicate)
    # ------------------------------------------------------------------
    print("\n=== Missing (key, model) combos per model ===")
    print(missing[MODEL_COL].value_counts())

    print("\n=== Duplicate (key, model) combos per model ===")
    if not dup_combos.empty:
        print(dup_combos[MODEL_COL].value_counts())
    else:
        print("No per-key duplicates found.")

    # ------------------------------------------------------------------
    # 4) Build a CLEAN dataset:
    #    Only keep image keys that have exactly ONE row per model,
    #    for *all* models in all_models.
    # ------------------------------------------------------------------

    # Valid combos = count == 1
    valid_combos = full[full["count"] == 1].copy()

    # For each key, how many models are present with exactly one row?
    per_key_valid = (
        valid_combos.groupby(KEY_COLS)[MODEL_COL]
        .nunique()
        .reset_index(name="n_unique_models")
    )

    # Good keys: have *all* models present exactly once
    good_keys = per_key_valid[per_key_valid["n_unique_models"] == num_models][KEY_COLS]

    # Filter original df to keep only rows belonging to good keys
    clean_df = df.merge(good_keys, on=KEY_COLS, how="inner")

    # Sanity check: after cleaning, each (key, model) should appear exactly once
    gm_clean = (
        clean_df.groupby(KEY_COLS + [MODEL_COL])
        .size()
        .reset_index(name="count")
    )
    assert gm_clean["count"].eq(1).all(), "Clean dataset still has duplicates!"

    # ------------------------------------------------------------------
    # 5) Save outputs
    # ------------------------------------------------------------------
    clean_df.to_csv(CLEAN_OUT, index=False)
    missing.to_csv(MISSING_OUT, index=False)
    dup_combos.to_csv(DUP_COMBOS_OUT, index=False)
    dup_rows.to_csv(DUP_ROWS_OUT, index=False)

    print(f"\n✅ Clean dataset saved to: {CLEAN_OUT}")
    print(f"   Rows in clean dataset: {len(clean_df)}")

    print(f"\n📁 Missing (key, model) combos saved to: {MISSING_OUT}")
    print(f"📁 Duplicate (key, model) combos (grouped) saved to: {DUP_COMBOS_OUT}")
    print(f"📁 Duplicate rows saved to: {DUP_ROWS_OUT}")


if __name__ == "__main__":
    main()
