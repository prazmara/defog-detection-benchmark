#!/usr/bin/env python3
"""
Correlation Analysis Tool for Human vs VLM Judge Scores

This script analyzes and visualizes the correlation between human judge scores
and VLM (Vision Language Model) judge scores for image quality assessment.

Usage:
    python create_correlation_graph.py [OPTIONS]
    
Examples:
    # Use default paths
    python create_correlation_graph.py
    
    # Specify custom paths
    python create_correlation_graph.py --human humanjudge/scores.csv --vlm results/acdc/combined_model_stats.csv --rubric results/original_per_model_rubric_stats.csv
    
    # Custom output prefix
    python create_correlation_graph.py --output my_correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import argparse
import sys
from pathlib import Path


def load_data(human_path, vlm_combined_path, vlm_rubric_path):
    """Load all required data files."""
    try:
        human_df = pd.read_csv(human_path)
        vlm_combined_df = pd.read_csv(vlm_combined_path)
        vlm_rubric_df = pd.read_csv(vlm_rubric_path)
        return human_df, vlm_combined_df, vlm_rubric_df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def extract_human_scores(human_df):
    """Extract total and category scores from human judge data."""
    if 'Category' in human_df.columns:
        totals = human_df[human_df['Category'] == 'TOTAL'][['Model', 'Mean']].copy()
        totals.columns = ['model', 'human_mean_total']
        rubrics = human_df[human_df['Category'] != 'TOTAL'].copy()
    else:
        totals = human_df.copy()
        rubrics = None
    
    return totals, rubrics


def match_models(human_totals, vlm_rubric_df, model_mapping=None):
    """Match models between human and VLM datasets."""
    if model_mapping is None:
        model_mapping = {
            'dehazformer_pretrained': 'dehazeformer',
            'dehazformer_trained': 'b01_dhft',
            'flux_split_cot': 'flux_split_non_cot',
        }
    
    matched_data = []
    
    for _, row in human_totals.iterrows():
        human_model = row['model']
        human_model_lower = human_model.lower()
        
        vlm_match = vlm_rubric_df[vlm_rubric_df['model'].str.lower() == human_model_lower]
        
        if vlm_match.empty and human_model in model_mapping:
            mapped_name = model_mapping[human_model]
            vlm_match = vlm_rubric_df[vlm_rubric_df['model'].str.lower() == mapped_name.lower()]
        
        if vlm_match.empty:
            for vlm_model in vlm_rubric_df['model']:
                if human_model_lower in vlm_model.lower() or vlm_model.lower() in human_model_lower:
                    vlm_match = vlm_rubric_df[vlm_rubric_df['model'] == vlm_model]
                    break
        
        if not vlm_match.empty:
            human_score = row['human_mean_total']
            
            vlm_total = (vlm_match['visibility_restoration_mean'].values[0] +
                        vlm_match['boundary_clarity_mean'].values[0] +
                        #vlm_match['scene_consistency_mean'].values[0] +
                        #vlm_match['object_consistency_mean'].values[0] +
                        vlm_match['perceived_detectability_mean'].values[0] 
                        #vlm_match['relation_consistency_mean'].values[0]
                        )
            
            matched_data.append({
                'model': human_model,
                'vlm_model': vlm_match['model'].values[0],
                'human_score': human_score,
                'vlm_score': vlm_total
            })
    
    return matched_data


def calculate_vlm_totals(vlm_rubric_df):
    """Calculate total scores for all VLM models."""
    vlm_totals = []
    for _, row in vlm_rubric_df.iterrows():
        total = (row['visibility_restoration_mean'] + 
                row['boundary_clarity_mean'] +
                row['scene_consistency_mean'] +
                row['object_consistency_mean'] +
                row['perceived_detectability_mean'] +
                row['relation_consistency_mean'])
        vlm_totals.append({
            'model': row['model'],
            'vlm_total': total
        })
    return pd.DataFrame(vlm_totals)


def create_visualizations(matched_data, human_totals, vlm_totals_df, human_rubrics, 
                         vlm_rubric_df, output_prefix):
    """Create individual correlation visualization PNG files."""
    
    output_base = Path(output_prefix).stem
    output_dir = Path(output_prefix).parent
    if output_dir == Path('.'):
        output_dir = Path.cwd()
    
    saved_files = []
    
    # Plot 1: Scatter plot of matched models
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    if matched_data:
        
        matched_df = pd.DataFrame(matched_data)
        ax1.scatter(matched_df['human_score'], matched_df['vlm_score'], 
                   s=150, alpha=0.6, color='steelblue', edgecolors='black', linewidth=1.5)
        
        for _, row in matched_df.iterrows():
            ax1.annotate(row['model'], (row['human_score'], row['vlm_score']), 
                        fontsize=10, alpha=0.8, xytext=(5, 5), textcoords='offset points')
        
        if len(matched_df) > 1:
            pearson_corr, pearson_p = pearsonr(matched_df['human_score'], 
                                              matched_df['vlm_score'])
            spearman_corr, spearman_p = spearmanr(matched_df['human_score'], 
                                                   matched_df['vlm_score'])
            
            z = np.polyfit(matched_df['human_score'], matched_df['vlm_score'], 1)
            p = np.poly1d(z)
            ax1.plot(matched_df['human_score'], p(matched_df['human_score']), 
                    "r--", alpha=0.5, linewidth=2, label='Linear fit')
            
            ax1.text(0.05, 0.95, 
                    f'Pearson r: {pearson_corr:.3f} (p={pearson_p:.3f})\n'
                    f'Spearman ρ: {spearman_corr:.3f} (p={spearman_p:.3f})',
                    transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Human Judge Total Score', fontsize=13, fontweight='bold')
        ax1.set_ylabel('VLM Judge Total Score', fontsize=13, fontweight='bold')
        ax1.set_title('Matched Models: Total Score Correlation', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'No matched models found', 
                ha='center', va='center', fontsize=14)
        ax1.set_title('Matched Models: Total Score Correlation', 
                     fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output1 = output_dir / f"{output_base}_scatter.png"
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(str(output1))
    
    # Plot 2: Distribution comparison
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    human_total_scores = human_totals['human_mean_total'].values
    print("+++++++++++++++++++++++++++++++++")
    #print(vlm_totals_df[matched_data])
    print(matched_data)
    print("+++++++++++++++++++++++++++++++++")
    human_totals_match_data = []
    vlm_total_match_data = []
    for i in matched_data:
        human_totals_match_data.append(i['human_score'])
        vlm_total_match_data.append(i['vlm_score'])

    vlm_total_scores = pd.DataFrame(vlm_total_match_data).values.flatten()
    
    positions = [1, 2]
    colors = ['tab:blue', 'tab:orange']   # different colors for each box
    print(human_total_scores)
    print(vlm_total_scores)
    bp = ax2.boxplot([human_total_scores, vlm_total_scores], positions=positions, 
                      widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      showfliers=False
                    #   medianprops=dict(color='red', linewidth=2)
                      )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    for median in bp['medians']:
        median.set_visible(False)
    ax2.set_xticklabels(['Human Judge', 'VLM Judge'], fontsize=12)
    ax2.set_ylabel('Total Score 0-15', fontsize=13, fontweight='bold')
    ax2.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    

    # --- Show individual judgments as jittered dots ---
    jitter = 0.002
    x_positions = [1, 2]

    ax2.scatter(
        np.random.normal(x_positions[0], jitter, size=len(human_total_scores)),
        human_total_scores,
        alpha=0.8,
        s=30,
        edgecolor='k',
        linewidth=0.5,
        color='red'
    )

    ax2.scatter(
        np.random.normal(x_positions[1], jitter, size=len(vlm_total_scores)),
        vlm_total_scores,
        alpha=0.8,
        s=30,
        edgecolor='k',
        color='red',
        linewidth=0.5)



    # means = [human_total_scores.mean(), vlm_total_scores.mean()]
    # ax2.scatter(positions, means, color='green', s=150, zorder=3, 
    #            label='Mean', marker='D', edgecolors='black', linewidth=1.5)
    # ax2.legend(fontsize=11)
    
    human_stats_text = (f'μ={human_total_scores.mean():.2f}, '
                 f'σ={human_total_scores.std():.2f}\n')
    vlm_stats_text = (
                 f'μ={vlm_total_scores.mean():.2f}, '
                 f'σ={vlm_total_scores.std():.2f}')
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, mutation_aspect=2)
    # Left box (Human)
    ax2.text(
        0.35, 0.02, human_stats_text,
        transform=ax2.transAxes,
        fontsize=11,
        ha='right',     # LEFT side of the pair
        va='bottom',
        bbox=bbox_props
    )

    # Right box (VLM)
    ax2.text(
        0.65, 0.02,  vlm_stats_text ,
        transform=ax2.transAxes,
        fontsize=11,
        ha='left',      # RIGHT side of the pair
        va='bottom',
        bbox=bbox_props
    )

    
    plt.tight_layout()
    output2 = output_dir / f"{output_base}_distribution.png"
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(str(output2))
    
    # Plot 3: Category-level correlations
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    if matched_data and human_rubrics is not None:
        category_mapping = {
            'Visibility Restoration': 'visibility_restoration_mean',
            'Boundary Clarity': 'boundary_clarity_mean',
            'Scene Consistency': 'scene_consistency_mean',
            'Object Consistency': 'object_consistency_mean',
            'Perceived Detectability': 'perceived_detectability_mean',
            'Relation Consistency': 'relation_consistency_mean'
        }
        
        category_correlations = []
        
        for category, vlm_col in category_mapping.items():
            human_cat_scores = []
            vlm_cat_scores = []
            
            for match in matched_data:
                human_model = match['model']
                vlm_model = match['vlm_model']
                
                human_score = human_rubrics[
                    (human_rubrics['Model'] == human_model) & 
                    (human_rubrics['Category'] == category)
                ]['Mean'].values
                
                if len(human_score) > 0:
                    human_cat_scores.append(human_score[0])
                    vlm_score = vlm_rubric_df[vlm_rubric_df['model'] == vlm_model][vlm_col].values[0]
                    vlm_cat_scores.append(vlm_score)
            
            if len(human_cat_scores) > 1:
                corr, p_val = pearsonr(human_cat_scores, vlm_cat_scores)
                category_correlations.append({
                    'category': category.replace(' ', '\n'),
                    'correlation': corr,
                    'p_value': p_val
                })
        
        if category_correlations:
            cat_df = pd.DataFrame(category_correlations)
            colors = ['green' if p < 0.05 else 'orange' for p in cat_df['p_value']]
            bars = ax3.barh(cat_df['category'], cat_df['correlation'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_xlabel('Pearson Correlation Coefficient', fontsize=13, fontweight='bold')
            ax3.set_title('Category-Level Correlations\n(Green: p<0.05, Orange: p≥0.05)', 
                         fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlim(-1, 1)
            ax3.grid(True, alpha=0.3, axis='x')
            
            for bar, corr in zip(bars, cat_df['correlation']):
                width = bar.get_width()
                ax3.text(width + 0.02 if width > 0 else width - 0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{corr:.3f}', ha='left' if width > 0 else 'right', 
                        va='center', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for category correlation', 
                    ha='center', va='center', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No matched models for category analysis', 
                ha='center', va='center', fontsize=12)
    
    ax3.set_title('Category-Level Correlations', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output3 = output_dir / f"{output_base}_categories.png"
    plt.savefig(output3, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(str(output3))
    
    # Plot 4: Model comparison bar chart
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    human_models_list = human_totals['model'].tolist()
    human_scores_list = human_totals['human_mean_total'].tolist()
    
    x_pos = np.arange(len(human_models_list))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, human_scores_list, width, 
                   label='Human Judge', alpha=0.8, color='steelblue', 
                   edgecolor='black', linewidth=1.5)
    
    vlm_scores_list = []
    for model in human_models_list:
        match = next((m for m in matched_data if m['model'] == model), None)
        vlm_scores_list.append(match['vlm_score'] if match else 0)
    
    bars2 = ax4.bar(x_pos + width/2, vlm_scores_list, width,
                   label='VLM Judge', alpha=0.8, color='coral',
                   edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Total Score', fontsize=13, fontweight='bold')
    ax4.set_title('Score Comparison by Model', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(human_models_list, rotation=45, ha='right', fontsize=11)
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output4 = output_dir / f"{output_base}_comparison.png"
    plt.savefig(output4, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(str(output4))
    
    return saved_files


def print_summary(matched_data, human_totals, vlm_combined_df):
    """Print summary statistics and correlation metrics."""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    if matched_data:
        matched_df = pd.DataFrame(matched_data)
        print(f"\nMatched Models: {len(matched_df)}")
        print("\nModel Comparison Table:")
        print(matched_df.to_string(index=False))
        
        if len(matched_df) > 1:
            pearson_corr, pearson_p = pearsonr(matched_df['human_score'], 
                                              matched_df['vlm_score'])
            spearman_corr, spearman_p = spearmanr(matched_df['human_score'], 
                                                   matched_df['vlm_score'])
            print(f"\nOverall Correlation Metrics:")
            print(f"  Pearson correlation:  r = {pearson_corr:.4f} (p = {pearson_p:.4f})")
            print(f"  Spearman correlation: ρ = {spearman_corr:.4f} (p = {spearman_p:.4f})")
            
            if pearson_p < 0.05:
                print(f"  ✓ Significant correlation detected (p < 0.05)")
            else:
                print(f"  ✗ No significant correlation (p ≥ 0.05)")
    else:
        print("\nNo direct model matches found between human and VLM judges.")
        print("This suggests different model sets were evaluated.")
        print("\nHuman judge evaluated:", list(human_totals['model']))
        print("VLM judge evaluated:", list(vlm_combined_df['model']))
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between human and VLM judge scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python create_correlation_graph.py
  
  # Specify custom paths
  python create_correlation_graph.py --human my_human_data.csv --vlm my_vlm_data.csv
  
  # Custom output prefix
  python create_correlation_graph.py --output my_analysis
        """
    )
    
    parser.add_argument('--human', type=str, 
                       default='humanjudge1214/summary_statistics.csv',
                       help='Path to human judge CSV file (default: humanjudge/summary_statistics.csv)')
    
    parser.add_argument('--vlm', '--vlm-combined', type=str, 
                       default='cityscape_results/combined_per_model_total_stats.csv',
                       dest='vlm_combined',
                       help='Path to VLM combined stats CSV (default: results/acdc/combined_model_stats.csv)')
    
    parser.add_argument('--rubric', '--vlm-rubric', type=str,
                       default='cityscape_results/combined_per_model_rubric_stats.csv',
                       dest='vlm_rubric',
                       help='Path to VLM rubric stats CSV (default: results/original_per_model_rubric_stats.csv)')
    
    parser.add_argument('--output', '-o', type=str,
                       default='correlation_analysis',
                       help='Output filename prefix (default: correlation_analysis)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed progress information')
    
    args = parser.parse_args()
    
    # Verify input files exist
    for path, name in [(args.human, 'Human judge'), 
                       (args.vlm_combined, 'VLM combined'),
                       (args.vlm_rubric, 'VLM rubric')]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)
    
    if args.verbose:
        print("Loading data files...")
    
    # Load data
    human_df, vlm_combined_df, vlm_rubric_df = load_data(
        args.human, args.vlm_combined, args.vlm_rubric
    )
    
    if args.verbose:
        print(f"\nLoaded {len(human_df)} human judge records")
        print(f"Loaded {len(vlm_combined_df)} VLM combined records")
        print(f"Loaded {len(vlm_rubric_df)} VLM rubric records")
    
    # Extract scores
    human_totals, human_rubrics = extract_human_scores(human_df)
    
    if args.verbose:
        print(f"\nHuman judge models: {list(human_totals['model'])}")
        print(f"VLM rubric models: {list(vlm_rubric_df['model'])}")
    
    # Match models
    matched_data = match_models(human_totals, vlm_rubric_df)
    
    if args.verbose:
        print(f"\nMatched {len(matched_data)} models")
        for match in matched_data:
            print(f"  {match['model']} → {match['vlm_model']}")
    
    # Calculate VLM totals
    vlm_totals_df = calculate_vlm_totals(vlm_rubric_df)
    
    # Create visualizations
    saved_files = create_visualizations(matched_data, human_totals, vlm_totals_df, 
                                       human_rubrics, vlm_rubric_df, args.output)
    
    print(f"\n{'='*60}")
    print("GENERATED VISUALIZATIONS")
    print(f"{'='*60}")
    for i, filepath in enumerate(saved_files, 1):
        print(f"{i}. {filepath}")
    
    # Print summary
    print_summary(matched_data, human_totals, vlm_combined_df)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
