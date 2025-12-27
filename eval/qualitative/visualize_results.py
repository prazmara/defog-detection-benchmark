#!/usr/bin/env python3
"""
CSV Results Visualizer

Generate visualizations from post-processed rubric data CSV files.
Creates bar charts, comparison plots, and other visualizations.

Usage Examples:
    # Visualize combined stats
    python visualize_results.py --input results/acdc/combined_model_stats.csv --output results/acdc/plots/
    
    # Visualize combined rubric stats
    python visualize_results.py --input results/acdc/combined_model_rubric_stats.csv --output results/acdc/plots/ --type rubric
    
    # Visualize both
    python visualize_results.py --stats_csv results/acdc/combined_model_stats.csv --rubric_csv results/acdc/combined_model_rubric_stats.csv --output results/acdc/plots/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_plot_style():
    """Set up matplotlib and seaborn styling."""
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_total_scores(df: pd.DataFrame, output_dir: str, show_plot: bool = False):
    """
    Create bar plot of total scores with error bars.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: model, mean_total, std_total, sem_total
    output_dir : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    """
    # Filter out rows with missing data
    df_clean = df.dropna(subset=['mean_total', 'sem_total'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot with error bars
    x = np.arange(len(df_clean))
    bars = ax.bar(x, df_clean['mean_total'], yerr=df_clean['sem_total'], 
                   capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_clean)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Total Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - Total Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_clean['model'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_clean.iterrows()):
        ax.text(i, row['mean_total'] + row['sem_total'] + 0.1, 
                f"{row['mean_total']:.2f}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'total_scores_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rubric_heatmap(df: pd.DataFrame, output_dir: str, show_plot: bool = False):
    """
    Create heatmap of rubric category means.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rubric statistics
    output_dir : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    """
    # Extract mean columns
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    
    # Filter out rows with missing data
    df_clean = df[['model'] + mean_cols].dropna()
    
    # Create matrix for heatmap
    rubric_names = [col.replace('_mean', '').replace('_', ' ').title() 
                    for col in mean_cols]
    
    heatmap_data = df_clean[mean_cols].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(rubric_names)))
    ax.set_yticks(np.arange(len(df_clean)))
    ax.set_xticklabels(rubric_names, rotation=45, ha='right')
    ax.set_yticklabels(df_clean['model'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Score', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(df_clean)):
        for j in range(len(mean_cols)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Rubric Category Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rubric_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rubric_radar(df: pd.DataFrame, output_dir: str, show_plot: bool = False):
    """
    Create radar chart comparing models across rubric categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rubric statistics
    output_dir : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    """
    # Extract mean columns
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    
    # Filter out rows with missing data
    df_clean = df[['model'] + mean_cols].dropna()
    
    # Prepare data
    categories = [col.replace('_mean', '').replace('_', '\n').title() 
                  for col in mean_cols]
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df_clean)))
    
    for idx, (i, row) in enumerate(df_clean.iterrows()):
        values = row[mean_cols].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Rubric Category Comparison - Radar Chart', 
              size=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rubric_radar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_rubric_bars_grouped(df: pd.DataFrame, output_dir: str, show_plot: bool = False):
    """
    Create grouped bar chart for rubric categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with rubric statistics
    output_dir : str
        Directory to save the plot
    show_plot : bool
        Whether to display the plot
    """
    # Extract mean and sem columns
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    sem_cols = [col for col in df.columns if col.endswith('_sem')]
    
    # Filter out rows with missing data
    df_clean = df[['model'] + mean_cols + sem_cols].dropna()
    
    # Prepare data
    rubric_names = [col.replace('_mean', '').replace('_', ' ').title() 
                    for col in mean_cols]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(rubric_names))
    width = 0.8 / len(df_clean)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_clean)))
    
    for idx, (i, row) in enumerate(df_clean.iterrows()):
        means = [row[col] for col in mean_cols]
        sems = [row[col.replace('_mean', '_sem')] for col in mean_cols]
        
        offset = (idx - len(df_clean)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, 
                     label=row['model'], capsize=3, alpha=0.8,
                     color=colors[idx], edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Rubric Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by Rubric Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rubric_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rubric_grouped_bars.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(
        description='Visualize post-processed rubric data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--stats_csv',
        type=str,
        help='Path to combined_model_stats.csv file'
    )
    parser.add_argument(
        '--rubric_csv',
        type=str,
        help='Path to combined_model_rubric_stats.csv file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.stats_csv and not args.rubric_csv:
        parser.error("At least one of --stats_csv or --rubric_csv must be provided")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup plotting style
    setup_plot_style()
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {args.output}\n")
    
    # Generate total scores visualization
    if args.stats_csv:
        if os.path.exists(args.stats_csv):
            print(f"Processing total scores from: {args.stats_csv}")
            df_stats = pd.read_csv(args.stats_csv)
            plot_total_scores(df_stats, args.output, args.show)
            print()
        else:
            print(f"Warning: File not found: {args.stats_csv}\n")
    
    # Generate rubric visualizations
    if args.rubric_csv:
        if os.path.exists(args.rubric_csv):
            print(f"Processing rubric stats from: {args.rubric_csv}")
            df_rubric = pd.read_csv(args.rubric_csv)
            
            plot_rubric_heatmap(df_rubric, args.output, args.show)
            plot_rubric_radar(df_rubric, args.output, args.show)
            plot_rubric_bars_grouped(df_rubric, args.output, args.show)
            print()
        else:
            print(f"Warning: File not found: {args.rubric_csv}\n")
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
