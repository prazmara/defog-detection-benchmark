import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_logfile(logfile_path):
    """Parse the logfile and extract all metrics"""
    
    with open(logfile_path, 'r') as f:
        content = f.read()
    
    # Extract datasets and models
    datasets = []
    models = []
    
    # Find all dataset names
    dataset_pattern = r"Loading ([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*) dataset"
    datasets = list(set(re.findall(dataset_pattern, content)))
    
    # Find all model names
    model_pattern = r"Applying ([\w-]+-coco-torch) to"
    models = list(set(re.findall(model_pattern, content)))
    
    # Extract mAP values
    map_pattern = r"([\w-]+-coco-torch) on ([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*) - mAP = ([\d.]+)"
    map_matches = re.findall(map_pattern, content)
    
    # Extract class-wise performance
    class_perf_pattern = r"Class-wise performance for ([\w-]+-coco-torch) on ([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*):\s*\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (.*?)\n\d{4}-\d{2}-\d{2}"
    class_perf_matches = re.findall(class_perf_pattern, content, re.DOTALL)
    
    return {
        'datasets': datasets,
        'models': models,
        'map_data': map_matches,
        'class_perf_data': class_perf_matches
    }

def extract_processing_speeds(logfile_path):
    """Extract processing speeds from the log"""
    speeds = []
    
    with open(logfile_path, 'r') as f:
        content = f.read()
    
    # Group by model and dataset
    current_model = None
    current_dataset = None
    
    for line in content.split('\n'):
        if 'Applying' in line and 'dataset' in line:
            match = re.search(r"Applying ([\w-]+-coco-torch) to ([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*) dataset", line)
            if match:
                current_model = match.group(1)
                current_dataset = match.group(2)
        
        if 'samples/s' in line and current_model and current_dataset:
            speed_match = re.search(r"\[([\d.]+)m? elapsed, 0s remaining, ([\d.]+) samples/s\]", line)
            if speed_match:
                elapsed = speed_match.group(1)
                speed = float(speed_match.group(2))
                
                # Convert to minutes if needed
                if 'm' in elapsed:
                    elapsed = float(elapsed.replace('m', ''))
                else:
                    elapsed = float(elapsed) / 60  # Convert seconds to minutes
                
                speeds.append({
                    'model': current_model,
                    'dataset': current_dataset,
                    'elapsed_minutes': elapsed,
                    'samples_per_second': speed,
                    'total_samples': 500
                })
    
    return speeds

def create_processing_speed_analysis(speeds_df, save_dir):
    """Create processing speed analysis plots"""
    
    # 1. Average processing speed per model
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    avg_speeds = speeds_df.groupby('model')['samples_per_second'].mean().sort_values(ascending=False)
    
    bars1 = ax1.bar(range(len(avg_speeds)), avg_speeds.values, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Average Processing Speed (images/second)', fontsize=12)
    ax1.set_title('Average Processing Speed by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(avg_speeds)))
    ax1.set_xticklabels([model.replace('-coco-torch', '') for model in avg_speeds.index], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_avg_processing_speed.png'), dpi=300, bbox_inches='tight')
    # plt.show()

def create_map_analysis(map_df, save_dir):
    """Create mAP analysis plots"""
    
    # 1. mAP by model and dataset
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    pivot_map = map_df.pivot(index='dataset', columns='model', values='map')
    
    im1 = ax1.imshow(pivot_map.values, cmap='viridis', aspect='auto')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Dataset', fontsize=12)
    ax1.set_title('mAP Performance Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(pivot_map.columns)))
    ax1.set_xticklabels([model.replace('-coco-torch', '') for model in pivot_map.columns], rotation=45, ha='right')
    ax1.set_yticks(range(len(pivot_map.index)))
    ax1.set_yticklabels(pivot_map.index)
    
    # Remove grid lines
    ax1.grid(False)
    
    # Add text annotations with better visibility
    for i in range(len(pivot_map.index)):
        for j in range(len(pivot_map.columns)):
            value = pivot_map.iloc[i, j]
            # Use white text for dark backgrounds, black for light backgrounds
            text_color = 'white' if value < 0.5 else 'black'
            ax1.text(j, i, f'{value:.3f}',
                    ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
    
    plt.colorbar(im1, ax=ax1, label='mAP')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_map_heatmap.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    
    # 2. mAP bar graph by model and dataset
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get unique models and datasets
    models = pivot_map.columns
    datasets = pivot_map.index
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.8 / len(datasets)  # Adjust bar width based on number of datasets
    
    # Create bars for each dataset
    colors = plt.cm.Set3(np.linspace(0, 1, len(datasets)))  # Generate unique colors
    
    for i, dataset in enumerate(datasets):
        values = pivot_map.loc[dataset].values
        bars = ax2.bar(x + i * width, values, width, label=dataset, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.set_title('mAP Performance by Model and Dataset', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax2.set_xticklabels([model.replace('-coco-torch', '') for model in models], rotation=45, ha='right')
    ax2.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_map_bar_graph.png'), dpi=300, bbox_inches='tight')
    # plt.show()

def create_class_performance_analysis(class_data, save_dir):
    """Create class-wise performance analysis"""
    
    if not class_data:
        print("No class performance data found!")
        return
    
    # Create DataFrames for different metrics
    precision_data = []
    recall_data = []
    f1_data = []
    
    for match in class_data:
        model, dataset, perf_str = match
        
        try:
            # Parse the performance string
            perf_dict = eval(perf_str)
            
            for class_name, metrics in perf_dict.items():
                if class_name not in ['micro avg', 'macro avg', 'weighted avg']:
                    precision_data.append({
                        'model': model,
                        'dataset': dataset,
                        'class': class_name,
                        'precision': metrics.get('precision', 0)
                    })
                    
                    recall_data.append({
                        'model': model,
                        'dataset': dataset,
                        'class': class_name,
                        'recall': metrics.get('recall', 0)
                    })
                    
                    f1_data.append({
                        'model': model,
                        'dataset': dataset,
                        'class': class_name,
                        'f1_score': metrics.get('f1-score', 0)
                    })
        except:
            continue
    
    precision_df = pd.DataFrame(precision_data)
    recall_df = pd.DataFrame(recall_data)
    f1_df = pd.DataFrame(f1_data)
    
    # 1. Average precision by class and model
    if not precision_df.empty:
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        precision_pivot = precision_df.pivot_table(
            values='precision', 
            index='class', 
            columns='model', 
            aggfunc='mean'
        )
        
        im1 = ax1.imshow(precision_pivot.values, cmap='Blues', aspect='auto')
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Class', fontsize=12)
        ax1.set_title('Average Precision by Class and Model', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(precision_pivot.columns)))
        ax1.set_xticklabels([model.replace('-coco-torch', '') for model in precision_pivot.columns], rotation=45, ha='right')
        ax1.set_yticks(range(len(precision_pivot.index)))
        ax1.set_yticklabels(precision_pivot.index)
        
        # Remove grid lines
        ax1.grid(False)
        
        # Add text annotations with better visibility
        for i in range(len(precision_pivot.index)):
            for j in range(len(precision_pivot.columns)):
                value = precision_pivot.iloc[i, j]
                # Use white text for dark backgrounds, black for light backgrounds
                text_color = 'white' if value < 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}',
                        ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
        
        plt.colorbar(im1, ax=ax1, label='Precision')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '04_precision_heatmap.png'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    # 2. Average recall by class and model
    if not recall_df.empty:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        recall_pivot = recall_df.pivot_table(
            values='recall', 
            index='class', 
            columns='model', 
            aggfunc='mean'
        )
        
        im2 = ax2.imshow(recall_pivot.values, cmap='Greens', aspect='auto')
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_title('Average Recall by Class and Model', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(recall_pivot.columns)))
        ax2.set_xticklabels([model.replace('-coco-torch', '') for model in recall_pivot.columns], rotation=45, ha='right')
        ax2.set_yticks(range(len(recall_pivot.index)))
        ax2.set_yticklabels(recall_pivot.index)
        
        # Remove grid lines
        ax2.grid(False)
        
        # Add text annotations with better visibility
        for i in range(len(recall_pivot.index)):
            for j in range(len(recall_pivot.columns)):
                value = recall_pivot.iloc[i, j]
                # Use white text for dark backgrounds, black for light backgrounds
                text_color = 'white' if value < 0.5 else 'black'
                ax2.text(j, i, f'{value:.2f}',
                        ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
        
        plt.colorbar(im2, ax=ax2, label='Recall')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '05_recall_heatmap.png'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    # 3. Average F1-score by class and model
    if not f1_df.empty:
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        f1_pivot = f1_df.pivot_table(
            values='f1_score', 
            index='class', 
            columns='model', 
            aggfunc='mean'
        )
        
        im3 = ax3.imshow(f1_pivot.values, cmap='Reds', aspect='auto')
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('Class', fontsize=12)
        ax3.set_title('Average F1-Score by Class and Model', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(f1_pivot.columns)))
        ax3.set_xticklabels([model.replace('-coco-torch', '') for model in f1_pivot.columns], rotation=45, ha='right')
        ax3.set_yticks(range(len(f1_pivot.index)))
        ax3.set_yticklabels(f1_pivot.index)
        
        # Remove grid lines
        ax3.grid(False)
        
        # Add text annotations with better visibility
        for i in range(len(f1_pivot.index)):
            for j in range(len(f1_pivot.columns)):
                value = f1_pivot.iloc[i, j]
                # Use white text for dark backgrounds, black for light backgrounds
                text_color = 'white' if value < 0.5 else 'black'
                ax3.text(j, i, f'{value:.2f}',
                        ha="center", va="center", color=text_color, fontweight='bold', fontsize=11)
        
        plt.colorbar(im3, ax=ax3, label='F1-Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '06_f1_score_heatmap.png'), dpi=300, bbox_inches='tight')
        # plt.show()

def create_summary_statistics(speeds_df, map_df, class_data):
    """Create summary statistics and insights"""
    
    print("=" * 60)
    print("OBJECT DETECTION MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Processing speed summary
    print("\n1. PROCESSING SPEED ANALYSIS")
    print("-" * 30)
    
    avg_speeds = speeds_df.groupby('model')['samples_per_second'].mean().sort_values(ascending=False)
    print("Average processing speed by model:")
    for model, speed in avg_speeds.items():
        print(f"  {model.replace('-coco-torch', '')}: {speed:.1f} images/second")
    
    fastest_model = avg_speeds.index[0]
    slowest_model = avg_speeds.index[-1]
    print(f"\nFastest model: {fastest_model.replace('-coco-torch', '')} ({avg_speeds.iloc[0]:.1f} img/s)")
    print(f"Slowest model: {slowest_model.replace('-coco-torch', '')} ({avg_speeds.iloc[-1]:.1f} img/s)")
    
    # mAP summary
    print("\n2. mAP PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    avg_map = map_df.groupby('model')['map'].mean().sort_values(ascending=False)
    print("Average mAP by model:")
    for model, map_val in avg_map.items():
        print(f"  {model.replace('-coco-torch', '')}: {map_val:.3f}")
    
    best_model = avg_map.index[0]
    worst_model = avg_map.index[-1]
    print(f"\nBest performing model: {best_model.replace('-coco-torch', '')} (mAP: {avg_map.iloc[0]:.3f})")
    print(f"Worst performing model: {worst_model.replace('-coco-torch', '')} (mAP: {avg_map.iloc[-1]:.3f})")
    
    # Dataset performance
    dataset_avg = map_df.groupby('dataset')['map'].mean().sort_values(ascending=False)
    print(f"\nDataset difficulty (average mAP):")
    for dataset, map_val in dataset_avg.items():
        print(f"  {dataset}: {map_val:.3f}")
    
    # Class performance summary
    if class_data:
        print("\n3. CLASS PERFORMANCE ANALYSIS")
        print("-" * 30)
        
        # Extract class performance data
        class_perf = []
        for match in class_data:
            model, dataset, perf_str = match
            try:
                perf_dict = eval(perf_str)
                for class_name, metrics in perf_dict.items():
                    if class_name not in ['micro avg', 'macro avg', 'weighted avg']:
                        class_perf.append({
                            'class': class_name,
                            'f1_score': metrics.get('f1-score', 0),
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0)
                        })
            except:
                continue
        
        if class_perf:
            class_df = pd.DataFrame(class_perf)
            class_avg = class_df.groupby('class')[['f1_score', 'precision', 'recall']].mean().sort_values('f1_score', ascending=False)
            
            print("Class performance (average F1-score):")
            for class_name, row in class_avg.iterrows():
                print(f"  {class_name}: {row['f1_score']:.3f} (P: {row['precision']:.3f}, R: {row['recall']:.3f})")
            
            best_class = class_avg.index[0]
            worst_class = class_avg.index[-1]
            print(f"\nBest performing class: {best_class} (F1: {class_avg.iloc[0]['f1_score']:.3f})")
            print(f"Worst performing class: {worst_class} (F1: {class_avg.iloc[-1]['f1_score']:.3f})")
    
    # Speed vs Accuracy trade-off
    print("\n4. SPEED vs ACCURACY TRADE-OFF")
    print("-" * 30)
    
    # Merge speed and accuracy data
    speed_accuracy = speeds_df.merge(map_df, on=['model', 'dataset'])
    speed_accuracy_avg = speed_accuracy.groupby('model')[['samples_per_second', 'map']].mean()
    
    print("Model efficiency (Speed vs Accuracy):")
    for model, row in speed_accuracy_avg.iterrows():
        efficiency = row['map'] * row['samples_per_second']  # Simple efficiency metric
        print(f"  {model.replace('-coco-torch', '')}: {row['samples_per_second']:.1f} img/s, mAP: {row['map']:.3f}, Efficiency: {efficiency:.2f}")
    
    most_efficient = speed_accuracy_avg['map'] * speed_accuracy_avg['samples_per_second']
    most_efficient_model = most_efficient.idxmax()
    print(f"\nMost efficient model: {most_efficient_model.replace('-coco-torch', '')} (Efficiency: {most_efficient.max():.2f})")

def main():
    """Main function to run the analysis"""
    
    print("Analyzing logfile.txt...")
    
    # Create loganalysis directory
    save_dir = 'loganalysis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Parse the logfile
    parsed_data = parse_logfile('logfile.txt')
    processing_speeds = extract_processing_speeds('logfile.txt')
    
    # Create DataFrames
    speeds_df = pd.DataFrame(processing_speeds)
    map_df = pd.DataFrame(parsed_data['map_data'], columns=['model', 'dataset', 'map'])
    map_df['map'] = map_df['map'].astype(float)
    
    print(f"Found {len(parsed_data['datasets'])} datasets: {parsed_data['datasets']}")
    print(f"Found {len(parsed_data['models'])} models: {[model.replace('-coco-torch', '') for model in parsed_data['models']]}")
    print(f"Total evaluations: {len(parsed_data['map_data'])}")
    
    # Create all visualizations
    print("\nCreating all visualizations...")
    
    # Processing speed analysis
    create_processing_speed_analysis(speeds_df, save_dir)
    
    # mAP analysis
    create_map_analysis(map_df, save_dir)
    
    # Class performance analysis
    create_class_performance_analysis(parsed_data['class_perf_data'], save_dir)
    
    # Print summary statistics
    create_summary_statistics(speeds_df, map_df, parsed_data['class_perf_data'])

if __name__ == "__main__":
    main()
