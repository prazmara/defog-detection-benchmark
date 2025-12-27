#!/usr/bin/env python3
"""
Main pipeline runner for post-processing rubric data.

This script provides a command-line interface to run the post-processing pipeline
with customizable arguments. Can process a single CSV, multiple CSVs, or all CSVs in a folder.

Usage Examples:
    # Process single CSV with default settings
    python run_post_process_pipeline.py --input_csv db/clean_up/dataset_no_duplicate_rows.csv
    
    # Process multiple CSV files
    python run_post_process_pipeline.py --input_csv file1.csv file2.csv file3.csv --output_dir results/batch
    
    # Process all CSVs in a folder
    python run_post_process_pipeline.py --input_folder db/clean_up --output_dir results/folder_batch
    
    # Process folder with custom rubric columns
    python run_post_process_pipeline.py --input_folder db/clean_up --rubric_cols visibility_restoration boundary_clarity
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import List
from scripts.postprocess.post_process_data import pipeline


def get_csv_files_from_folder(folder_path: str) -> List[str]:
    """
    Get all CSV files from a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing CSV files
    
    Returns:
    --------
    List[str]
        List of CSV file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    
    return [str(f) for f in csv_files]


def process_multiple_csvs(
    csv_files: List[str],
    output_dir: str,
    rubric_cols: List[str] = None,
    verbose: bool = True,
    save_results: bool = True
):
    """
    Process multiple CSV files through the pipeline.
    
    Parameters:
    -----------
    csv_files : List[str]
        List of CSV file paths to process
    output_dir : str
        Base output directory for results
    rubric_cols : List[str], optional
        List of rubric column names
    verbose : bool
        Whether to print statistics
    save_results : bool
        Whether to save results to CSV files
    """
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        csv_name = Path(csv_file).stem
        csv_output_dir = os.path.join(output_dir, csv_name)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(csv_files)}: {csv_file}")
            print(f"{'='*80}")
        
        try:
            df, counts, stats_total, rubric_stats = pipeline(
                input_csv=csv_file,
                output_dir=csv_output_dir,
                rubric_cols=rubric_cols,
                verbose=verbose,
                save_results=save_results
            )
            results.append({
                'file': csv_file,
                'df': df,
                'counts': counts,
                'stats_total': stats_total,
                'rubric_stats': rubric_stats
            })
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    return results


def main():
    """
    Run the post-processing pipeline with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Post-process rubric data from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV
  %(prog)s --input_csv data.csv --output_dir results

  # Multiple CSVs
  %(prog)s --input_csv file1.csv file2.csv file3.csv

  # All CSVs in folder
  %(prog)s --input_folder db/clean_up --output_dir results/batch
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_csv',
        nargs='+',
        help='Path(s) to input CSV file(s). Can specify multiple files.'
    )
    input_group.add_argument(
        '--input_folder',
        type=str,
        help='Path to folder containing CSV files. Will process all CSV files in the folder.'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output CSV files (default: results/non_repeat)'
    )
    
    # Rubric columns
    parser.add_argument(
        '--rubric_cols',
        nargs='+',
        default=None,
        help='List of rubric column names. If not specified, uses default columns.'
    )
    
    # Flags
    parser.add_argument(
        '--no-verbose',
        dest='verbose',
        action='store_false',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--no-save',
        dest='save_results',
        action='store_false',
        help='Do not save results to CSV files'
    )
    
    args = parser.parse_args()
    
    # Get list of CSV files to process
    if args.input_csv:
        csv_files = args.input_csv
    else:  # input_folder
        csv_files = get_csv_files_from_folder(args.input_folder)
    
    # Process the CSV files
    if len(csv_files) == 1:
        # Single file processing
        print(f"Processing single CSV: {csv_files[0]}")
        df, counts, stats_total, rubric_stats = pipeline(
            input_csv=csv_files[0],
            output_dir=args.output_dir,
            rubric_cols=args.rubric_cols,
            verbose=args.verbose,
            save_results=args.save_results
        )
        print("\nPipeline completed successfully!")
    else:
        # Multiple files processing
        print(f"Processing {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {f}")
        
        results = process_multiple_csvs(
            csv_files=csv_files,
            output_dir=args.output_dir,
            rubric_cols=args.rubric_cols,
            verbose=args.verbose,
            save_results=args.save_results
        )
        
        print(f"\n{'='*80}")
        print(f"Pipeline completed! Processed {len(results)}/{len(csv_files)} files successfully.")
        print(f"{'='*80}")
        
        # Create combined statistics CSVs
        if results and args.save_results:
            # Combined total stats
            combined_stats = []
            for result in results:
                combined_stats.append(result['stats_total'])
            
            if combined_stats:
                combined_df = pd.concat(combined_stats, ignore_index=True)
                combined_output_path = os.path.join(args.output_dir, "combined_model_stats.csv")
                combined_df.to_csv(combined_output_path, index=False)
                print(f"\nCombined total statistics saved to: {combined_output_path}")
            
            # Combined rubric stats
            combined_rubric_stats = []
            for result in results:
                combined_rubric_stats.append(result['rubric_stats'])
            
            if combined_rubric_stats:
                combined_rubric_df = pd.concat(combined_rubric_stats, ignore_index=True)
                combined_rubric_output_path = os.path.join(args.output_dir, "combined_model_rubric_stats.csv")
                combined_rubric_df.to_csv(combined_rubric_output_path, index=False)
                print(f"Combined rubric statistics saved to: {combined_rubric_output_path}")


if __name__ == "__main__":
    main()


'''
# Process single CSV
python run_post_process_pipeline.py --input_csv db/clean_up/dataset_no_duplicate_rows.csv

# Process multiple CSVs
python run_post_process_pipeline.py --input_csv file1.csv file2.csv file3.csv --output_dir results/batch

# Process all CSVs in a folder
python run_post_process_pipeline.py --input_folder db/clean_up --output_dir results/folder_batch

# With custom rubric columns
python run_post_process_pipeline.py --input_folder db/clean_up --rubric_cols visibility_restoration boundary_clarity


'''
