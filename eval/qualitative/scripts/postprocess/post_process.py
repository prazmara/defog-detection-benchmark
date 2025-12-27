import pandas as pd

def process_results(csv_path: str):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Define rubric columns
    rubric_cols = [
        "visibility_restoration",
        "visual_distortion",
        "boundary_clarity",
        "scene_consistency",
        "fidelity",
        "realism",
        "total"
    ]

    # Compute averages grouped by model
    avg_scores = (
        df.groupby("model")[rubric_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    print("\n=== Average Scores by Model ===\n")
    print(avg_scores.to_string(index=False))

    return avg_scores
