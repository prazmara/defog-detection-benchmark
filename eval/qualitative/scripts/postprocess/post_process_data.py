import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional


def process_rubric_data(
    df: pd.DataFrame,
    rubric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process rubric data and compute statistics per model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with rubric columns and model information
    rubric_cols : List[str]
        List of rubric column names to process
    
    Returns:
    --------
    Tuple containing:
        - df: DataFrame with added total_score column
        - counts: Per-model image counts
        - stats_total: Per-model total score statistics (mean, std, sem)
        - rubric_stats: Per-model rubric statistics (mean, std, sem for each rubric)
    """
    # Convert rubric columns to numeric
    for c in rubric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Per-image total
    df["total_score"] = df[rubric_cols].sum(axis=1)
    
    # Per-model counts
    counts = df.groupby("model")["basename"].nunique().reset_index(name="n_images")
    
    # Per-model total stats
    stats_total = (
        df.groupby("model")["total_score"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={"mean":"mean_total", "std":"std_total", "count":"n"})
    )
    stats_total["sem_total"] = stats_total["std_total"] / np.sqrt(stats_total["n"])
    
    # Per-rubric stats
    rubric_means = df.groupby("model")[rubric_cols].mean().add_suffix("_mean")
    rubric_stds  = df.groupby("model")[rubric_cols].std(ddof=1).add_suffix("_std")
    rubric_counts = df.groupby("model")[rubric_cols].count().iloc[:, :1]
    rubric_counts.columns = ["n"]
    rubric_stats = rubric_means.join(rubric_stds).join(rubric_counts)
    for c in rubric_cols:
        rubric_stats[f"{c}_sem"] = rubric_stats[f"{c}_std"] / np.sqrt(rubric_stats["n"])
    rubric_stats = rubric_stats.reset_index()
    
    return df, counts, stats_total, rubric_stats


def pipeline(
    input_csv: str = "db/clean_up/dataset_no_duplicate_rows.csv",
    output_dir: str = "results/non_repeat",
    rubric_cols: Optional[List[str]] = None,
    verbose: bool = True,
    save_results: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline to process rubric data from CSV and generate statistics.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file (default: "db/clean_up/dataset_no_duplicate_rows.csv")
    output_dir : str
        Directory to save output CSV files (default: "results/non_repeat")
    rubric_cols : Optional[List[str]]
        List of rubric column names. If None, uses default rubric columns.
    verbose : bool
        Whether to print statistics (default: True)
    save_results : bool
        Whether to save results to CSV files (default: True)
    
    Returns:
    --------
    Tuple containing:
        - df: DataFrame with added total_score column
        - counts: Per-model image counts
        - stats_total: Per-model total score statistics
        - rubric_stats: Per-model rubric statistics
    
    Example:
    --------
    >>> from scripts.postprocess.post_process_data import pipeline
    >>> df, counts, stats_total, rubric_stats = pipeline(
    ...     input_csv="data/my_dataset.csv",
    ...     output_dir="results/my_experiment",
    ...     verbose=True,
    ...     save_results=True
    ... )
    """
    # Default rubric columns
    if rubric_cols is None:
        rubric_cols = [
            "visibility_restoration",
            "boundary_clarity",
            "scene_consistency",
            "object_consistency",
            "perceived_detectability",
            "relation_consistency",
        ]
    
    # Read data
    df = pd.read_csv(input_csv, on_bad_lines="skip")
    
    # Process data
    df, counts, stats_total, rubric_stats = process_rubric_data(df, rubric_cols)
    
    # Print statistics if verbose
    if verbose:
        print("\nPer-model counts:\n", counts)
        print("\nPer-model total_score stats (mean/std/SEM):\n", stats_total)
        print("\nPer-model rubric stats (means/std/SEM):\n", rubric_stats)
    
    # Save results if requested
    if save_results:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        df.to_csv(f"{output_dir}/original_with_totals.csv", index=False)
        stats_total.to_csv(f"{output_dir}/original_per_model_total_stats.csv", index=False)
        rubric_stats.to_csv(f"{output_dir}/original_per_model_rubric_stats.csv", index=False)
        if verbose:
            print(f"\nResults saved to {output_dir}/")
    
    return df, counts, stats_total, rubric_stats
