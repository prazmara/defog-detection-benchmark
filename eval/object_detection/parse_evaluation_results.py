#!/usr/bin/env python3
"""
Script to parse evaluation results from outputseg.txt file.
Extracts method names and their corresponding panoptic segmentation metrics,
sorts by PQ (All row) from highest to lowest, and outputs to JSON.
"""

import re
import json
from typing import List, Dict, Any

def parse_evaluation_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the evaluation results file and extract method names with their metrics.
    
    Args:
        file_path: Path to the outputseg.txt file
        
    Returns:
        List of dictionaries containing method names and their metrics
    """
    results = []
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Pattern to match evaluation method and its results
    # Looks for "Evaluating method: <method_name>" followed by the metrics table
    pattern = r'Evaluating method: (\w+)\s*Evaluation panoptic segmentation metrics:.*?Time elapsed: [\d.]+ seconds'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    # For each method, extract the metrics table
    for method_name in matches:
        # Find the specific section for this method
        method_pattern = rf'Evaluating method: {re.escape(method_name)}.*?Time elapsed: [\d.]+ seconds'
        method_section = re.search(method_pattern, content, re.DOTALL)
        
        if method_section:
            section_text = method_section.group(0)
            
            # Extract the metrics table (lines with PQ, SQ, RQ, N)
            table_pattern = r'All\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
            table_match = re.search(table_pattern, section_text)
            
            if table_match:
                pq_all = float(table_match.group(1))
                sq_all = float(table_match.group(2))
                rq_all = float(table_match.group(3))
                n_all = int(table_match.group(4))
                
                # Also extract Things and Stuff metrics
                things_pattern = r'Things\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
                stuff_pattern = r'Stuff\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
                
                things_match = re.search(things_pattern, section_text)
                stuff_match = re.search(stuff_pattern, section_text)
                
                result = {
                    'method': method_name,
                    'metrics': {
                        'all': {
                            'PQ': pq_all,
                            'SQ': sq_all,
                            'RQ': rq_all,
                            'N': n_all
                        }
                    }
                }
                
                if things_match:
                    result['metrics']['things'] = {
                        'PQ': float(things_match.group(1)),
                        'SQ': float(things_match.group(2)),
                        'RQ': float(things_match.group(3)),
                        'N': int(things_match.group(4))
                    }
                
                if stuff_match:
                    result['metrics']['stuff'] = {
                        'PQ': float(stuff_match.group(1)),
                        'SQ': float(stuff_match.group(2)),
                        'RQ': float(stuff_match.group(3)),
                        'N': int(stuff_match.group(4))
                    }
                
                results.append(result)
    
    return results

def sort_by_pq_all(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort results by PQ (All row) from highest to lowest.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Sorted list of results
    """
    return sorted(results, key=lambda x: x['metrics']['all']['PQ'], reverse=True)

def print_results(results: List[Dict[str, Any]]) -> None:
    """
    Print the sorted results in a formatted table.
    
    Args:
        results: List of evaluation results
    """
    print("Evaluation Results Sorted by PQ (All) - Highest to Lowest")
    print("=" * 80)
    print(f"{'Rank':<4} {'Method':<25} {'PQ (All)':<8} {'SQ (All)':<8} {'RQ (All)':<8} {'N (All)':<6}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        method = result['method']
        metrics = result['metrics']['all']
        print(f"{i:<4} {method:<25} {metrics['PQ']:<8.1f} {metrics['SQ']:<8.1f} {metrics['RQ']:<8.1f} {metrics['N']:<6}")

def save_to_json(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to a JSON file.
    
    Args:
        results: List of evaluation results
        output_file: Output JSON file path
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

def main():
    """Main function to run the script."""
    input_file = "outputseg.txt"
    output_file = "evaluation_results.json"
    
    try:
        # Parse the evaluation file
        # print("Parsing evaluation results...")
        results = parse_evaluation_file(input_file)
        
        if not results:
            print("No evaluation results found!")
            return
        
        # Sort by PQ (All) from highest to lowest
        sorted_results = sort_by_pq_all(results)
        
        # Print results
        print_results(sorted_results)
        
        # Save to JSON
        save_to_json(sorted_results, output_file)
        
        print(f"\nTotal methods evaluated: {len(results)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
