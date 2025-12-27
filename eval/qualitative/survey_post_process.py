#!/usr/bin/env python3
"""
Survey Post-Processing Script
Analyzes human judge scores for defogging models across multiple categories.
Generates visualizations and comprehensive reports.
"""

import pandas as pd
import numpy as np
import os
from tools.visualization_utils import (
    setup_plot_style,
    plot_total_scores_bar,
    plot_grouped_bars,
    plot_heatmap,
    plot_radar_chart,
    generate_text_report
)

# Setup plotting style
setup_plot_style()

# add CLI args 
import argparse
parser = argparse.ArgumentParser(description='Post-process human judge scores for defogging models.')
parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing human judge scores.')
#output file 
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output files.')
args = parser.parse_args()


# Read the CSV
df = pd.read_csv(args.input_csv)

outputfolder = args.output_dir
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

# Categories used in the rubric
categories = [
    "Visibility Restoration",
    "Boundary Clarity",
    #"Perceived Detectability",
    #"Scene Consistency",
    "Object Consistency",
    #"Relation Consistency",
]

# Model patterns to identify in column names
# Note: Order matters! Check more specific patterns first
models = {
    "flux_best": "defoggedflux best",
    "dehazformer_trained": "dehazformer trained",
    "dehazformer_pretrained": "dehazformer",  # This will match columns WITHOUT "trained"
}

print("="*80)
print("SURVEY POST-PROCESSING ANALYSIS")
print("="*80)
print(f"\nTotal respondents: {len(df)}")
print(f"Total columns: {len(df.columns)}")

results = {}

for model_key, model_pattern in models.items():
    print(f"\n{'='*80}")
    print(f"Processing Model: {model_key}")
    print(f"Pattern: '{model_pattern}'")
    print('='*80)
    
    model_data = {}
    
    # For each category, find ALL columns for this model
    for cat in categories:
        # Find all columns matching this category and model
        if model_key == "dehazformer_pretrained":
            # Special handling: must have "dehazformer" but NOT "trained"
            matching_cols = [
                c for c in df.columns 
                if cat in c and "dehazformer" in c and "trained" not in c
            ]
        else:
            matching_cols = [
                c for c in df.columns 
                if cat in c and model_pattern in c
            ]
        
        print(f"  {cat}: found {len(matching_cols)} image columns")
        
        if len(matching_cols) == 0:
            print(f"    WARNING: No columns found!")
            continue
        
        # Get all values for this category across all images
        category_values = df[matching_cols].values.flatten()
        # Remove NaN values
        category_values = category_values[~np.isnan(category_values)]
        
        # Validate scores are in 0-5 range
        invalid_scores = category_values[(category_values < 0) | (category_values > 5)]
        if len(invalid_scores) > 0:
            print(f"    WARNING: Found {len(invalid_scores)} scores outside 0-5 range")
            print(f"    Invalid values: {invalid_scores}")
        
        # Clip scores to 0-5 range
        category_values = np.clip(category_values, 0, 5)
        
        model_data[cat] = {
            'columns': matching_cols,
            'values': category_values,
            'mean': np.mean(category_values),
            'std': np.std(category_values, ddof=1),  # Sample std
            'sem': np.std(category_values, ddof=1) / np.sqrt(len(category_values))
        }
    
    # Calculate total scores (sum of all 6 categories per image)
    # We need to organize columns by image first
    # Each image has 6 scores (one per category)
    
    # Group columns by image
    image_totals = []
    
    # Assuming all categories have the same images, use first category to get image list
    first_category_cols = list(model_data.values())[0]['columns']
    
    for img_col in first_category_cols:
        # Extract image identifier from column name
        # Get corresponding columns for all categories for this image
        img_scores = []
        
        for cat in categories:
            if cat in model_data:
                # Find the column for this category that matches this image
                matching_col = [c for c in model_data[cat]['columns'] if img_col.split(' – ')[1] in c]
                if matching_col:
                    score = df[matching_col[0]].iloc[0]  # Get score from first (only) respondent
                    if not np.isnan(score):
                        img_scores.append(score)
        
        # Sum all category scores for this image
        if len(img_scores) > 0:
            image_totals.append(sum(img_scores))
    
    image_totals = np.array(image_totals)
    
    # Store results
    results[model_key] = {
        'categories': model_data,
        'total_mean': np.mean(image_totals),
        'total_std': np.std(image_totals, ddof=1) if len(image_totals) > 1 else 0,
        'total_sem': np.std(image_totals, ddof=1) / np.sqrt(len(image_totals)) if len(image_totals) > 1 else 0,
        'total_count': len(image_totals)
    }

# Print summary report
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for model_key, model_results in results.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {model_key.upper()}")
    print('='*80)
    
    print(f"\n{'Category':<30} {'Mean':>8} {'Std':>8} {'SEM':>8} {'N':>8}")
    print('-'*80)
    
    for cat in categories:
        if cat in model_results['categories']:
            data = model_results['categories'][cat]
            print(f"{cat:<30} {data['mean']:>8.3f} {data['std']:>8.3f} {data['sem']:>8.3f} {len(data['values']):>8}")
    
    print('-'*80)
    print(f"{'TOTAL (All Categories)':<30} {model_results['total_mean']:>8.3f} "
          f"{model_results['total_std']:>8.3f} {model_results['total_sem']:>8.3f} "
          f"{model_results['total_count']:>8}")

# Create summary DataFrame for export
summary_data = []

for model_key, model_results in results.items():
    # Per-category rows
    for cat in categories:
        if cat in model_results['categories']:
            data = model_results['categories'][cat]
            summary_data.append({
                'Model': model_key,
                'Category': cat,
                'Mean': data['mean'],
                'Std': data['std'],
                'SEM': data['sem'],
                'N': len(data['values'])
            })
    
    # Total row
    summary_data.append({
        'Model': model_key,
        'Category': 'TOTAL',
        'Mean': model_results['total_mean'],
        'Std': model_results['total_std'],
        'SEM': model_results['total_sem'],
        'N': model_results['total_count']
    })

summary_df = pd.DataFrame(summary_data)

# Save to CSV
output_file = f"{outputfolder}/summary_statistics.csv"
summary_df.to_csv(output_file, index=False)
print(f"\n{'='*80}")
print(f"Summary saved to: {output_file}")
print('='*80)

# Create a pivot table for easier comparison
print("\n" + "="*80)
print("COMPARISON TABLE (MEAN SCORES)")
print("="*80)

pivot_mean = summary_df.pivot(index='Category', columns='Model', values='Mean')
print(pivot_mean.to_string(float_format=lambda x: f'{x:.3f}'))

print("\n" + "="*80)
print("COMPARISON TABLE (SEM)")
print("="*80)

pivot_sem = summary_df.pivot(index='Category', columns='Model', values='SEM')
print(pivot_sem.to_string(float_format=lambda x: f'{x:.3f}'))

# Save pivot tables
pivot_mean.to_csv(f"{outputfolder}/summary_means.csv")
pivot_sem.to_csv(f"{outputfolder}/summary_sem.csv")

print(f"\n{'='*80}")
print("Additional exports:")
print(f"  - humanjudge/summary_means.csv")
print(f"  - humanjudge/summary_sem.csv")
print('='*80)

# ============================================================================
# GENERATE VISUALIZATIONS AND REPORT
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS AND REPORTS")
print('='*80)

# Create output directories
plots_dir = f"{outputfolder}/plots"
os.makedirs(plots_dir, exist_ok=True)

model_names = list(results.keys())

# 1. Total scores bar chart
print("\n  Creating total scores bar chart...")
total_means = [results[m]['total_mean'] for m in model_names]
total_sems = [results[m]['total_sem'] for m in model_names]

plot_total_scores_bar(
    model_names=model_names,
    means=total_means,
    sems=total_sems,
    output_path=os.path.join(plots_dir, 'total_scores_comparison.png'),
    title='Human Survey - Total Scores Comparison',
    ylabel='Mean Total Score (Sum of 6 Categories)'
)
print(f"    ✓ Saved: {plots_dir}/total_scores_comparison.png")

# 2. Grouped bars for categories
print("  Creating category comparison bar chart...")
category_data = {}
for model_key in model_names:
    category_data[model_key] = {
        cat: results[model_key]['categories'][cat]
        for cat in categories
    }

plot_grouped_bars(
    model_names=model_names,
    categories=categories,
    data_dict=category_data,
    output_path=os.path.join(plots_dir, 'category_comparison.png'),
    title='Human Survey - Performance by Category'
)
print(f"    ✓ Saved: {plots_dir}/category_comparison.png")

# 3. Heatmap
print("  Creating heatmap...")
heatmap_data = np.array([
    [results[m]['categories'][cat]['mean'] for cat in categories]
    for m in model_names
])

plot_heatmap(
    model_names=model_names,
    categories=categories,
    data_matrix=heatmap_data,
    output_path=os.path.join(plots_dir, 'category_heatmap.png'),
    title='Human Survey - Category Performance Heatmap'
)
print(f"    ✓ Saved: {plots_dir}/category_heatmap.png")

# 4. Radar chart
print("  Creating radar chart...")
radar_data = {
    model_key: [results[model_key]['categories'][cat]['mean'] for cat in categories]
    for model_key in model_names
}

plot_radar_chart(
    model_names=model_names,
    categories=categories,
    data_dict=radar_data,
    output_path=os.path.join(plots_dir, 'radar_comparison.png'),
    title='Human Survey - Radar Chart Comparison'
)
print(f"    ✓ Saved: {plots_dir}/radar_comparison.png")

# 5. Generate text report
print("  Generating comprehensive text report...")
report_path = g=f"{outputfolder}/survey_report.txt"
generate_text_report(
    results=results,
    categories=categories,
    output_path=report_path,
    title="Human Survey Analysis Report - Defogging Models"
)
print(f"    ✓ Saved: {report_path}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print('='*80)
print("\nGenerated files:")
print("  CSV Files:")
print("    - humanjudge/summary_statistics.csv")
print("    - humanjudge/summary_means.csv")
print("    - humanjudge/summary_sem.csv")
print("  Visualizations:")
print("    - humanjudge/plots/total_scores_comparison.png")
print("    - humanjudge/plots/category_comparison.png")
print("    - humanjudge/plots/category_heatmap.png")
print("    - humanjudge/plots/radar_comparison.png")
print("  Reports:")
print("    - humanjudge/survey_report.txt")
print('='*80)
