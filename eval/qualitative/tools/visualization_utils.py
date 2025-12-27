#!/usr/bin/env python3
"""
Visualization Utilities
Reusable visualization functions for survey and model performance data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
import pandas as pd


def setup_plot_style():
    """Set up matplotlib styling."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_total_scores_bar(
    model_names: List[str],
    means: List[float],
    sems: List[float],
    output_path: str,
    title: str = "Model Performance Comparison - Total Scores",
    ylabel: str = "Mean Total Score"
):
    """
    Create bar chart of total scores with error bars.
    
    Parameters:
    -----------
    model_names : List[str]
        Names of models
    means : List[float]
        Mean scores for each model
    sems : List[float]
        Standard errors for each model
    output_path : str
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(i, mean + sem + 0.5, f"{mean:.1f}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_grouped_bars(
    model_names: List[str],
    categories: List[str],
    data_dict: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Model Performance by Category",
    ylabel: str = "Mean Score",
    include_error_bars: bool = True
):
    """
    Create grouped bar chart for multiple categories.
    
    Parameters:
    -----------
    model_names : List[str]
        Names of models
    categories : List[str]
        Category names
    data_dict : Dict[str, Dict[str, float]]
        Nested dict: {model: {category: {'mean': float, 'sem': float}}}
    output_path : str
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    include_error_bars : bool
        Whether to include error bars
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(categories))
    width = 0.8 / len(model_names)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    for idx, model_key in enumerate(model_names):
        means = [data_dict[model_key][cat]['mean'] for cat in categories]
        
        if include_error_bars:
            sems = [data_dict[model_key][cat]['sem'] for cat in categories]
        else:
            sems = None
            
        offset = (idx - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, label=model_key,
                      capsize=3, alpha=0.8, color=colors[idx], 
                      edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_heatmap(
    model_names: List[str],
    categories: List[str],
    data_matrix: np.ndarray,
    output_path: str,
    title: str = "Performance Heatmap",
    cmap: str = 'YlOrRd',
    vmin: float = 0,
    vmax: float = 5
):
    """
    Create heatmap visualization.
    
    Parameters:
    -----------
    model_names : List[str]
        Names of models (y-axis)
    categories : List[str]
        Category names (x-axis)
    data_matrix : np.ndarray
        2D array of shape (n_models, n_categories)
    output_path : str
        Path to save the plot
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Min and max values for colormap
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Score', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_radar_chart(
    model_names: List[str],
    categories: List[str],
    data_dict: Dict[str, List[float]],
    output_path: str,
    title: str = "Radar Chart Comparison"
):
    """
    Create radar/spider chart for comparing models across categories.
    
    Parameters:
    -----------
    model_names : List[str]
        Names of models
    categories : List[str]
        Category names
    data_dict : Dict[str, List[float]]
        Dict mapping model names to list of values for each category
    output_path : str
        Path to save the plot
    title : str
        Plot title
    """
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(model_names)))
    
    for idx, model_name in enumerate(model_names):
        values = data_dict[model_name]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title, size=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_text_report(
    results: Dict,
    categories: List[str],
    output_path: str,
    title: str = "Survey Analysis Report"
):
    """
    Generate a comprehensive text report.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary with model statistics
    categories : List[str]
        Category names
    output_path : str
        Path to save the report
    title : str
        Report title
    """
    from datetime import datetime
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for model_key, model_results in results.items():
            f.write("="*80 + "\n")
            f.write(f"MODEL: {model_key.upper()}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Category':<30} {'Mean':>8} {'Std':>8} {'SEM':>8} {'N':>8}\n")
            f.write("-"*80 + "\n")
            
            for cat in categories:
                if cat in model_results['categories']:
                    data = model_results['categories'][cat]
                    f.write(f"{cat:<30} {data['mean']:>8.3f} {data['std']:>8.3f} "
                           f"{data['sem']:>8.3f} {len(data['values']):>8}\n")
            
            f.write("-"*80 + "\n")
            f.write(f"{'TOTAL (All Categories)':<30} {model_results['total_mean']:>8.3f} "
                   f"{model_results['total_std']:>8.3f} {model_results['total_sem']:>8.3f} "
                   f"{model_results['total_count']:>8}\n")
            f.write("\n\n")
        
        # Add summary comparison
        f.write("="*80 + "\n")
        f.write("RANKING BY TOTAL SCORE\n")
        f.write("="*80 + "\n\n")
        
        ranked = sorted(results.items(), key=lambda x: x[1]['total_mean'], reverse=True)
        for rank, (model_key, model_results) in enumerate(ranked, 1):
            f.write(f"{rank}. {model_key}: {model_results['total_mean']:.2f} "
                   f"(±{model_results['total_sem']:.2f})\n")
    
    return output_path
