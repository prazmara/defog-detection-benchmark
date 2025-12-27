"""Data loading utilities for VLM Judge Pipeline."""

import argparse
import os
import random
from typing import List, Set

from tools.io import list_images
from tools.helpers import read_csv_and_add_png, read_missing_csv_and_find_file_name
from pipeline.config import CITYSCAPE_CITIES, DEFAULT_SAMPLE_SIZE, CANDIDATE_PATHS_FILE


def load_skip_basenames(csv_path: str) -> Set[str]:
    """Load basenames to skip from a CSV file."""
    skip_basenames = set()
    if not csv_path:
        return skip_basenames
    
    try:
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 1:
                    skip_basenames.add(parts[0])
    except FileNotFoundError:
        print(f"[WARN] Skip CSV not found: {csv_path}")
    
    return skip_basenames


def filter_by_skip_basenames(image_list: List[str], skip_basenames: Set[str]) -> List[str]:
    """Filter out images whose basenames are in the skip set."""
    if not skip_basenames:
        return image_list
    
    filtered = []
    for img_path in image_list:
        base = os.path.basename(img_path).rsplit(".", 1)[0]
        if base not in skip_basenames:
            filtered.append(img_path)
    
    return filtered


def load_reuse_samples(sample_file: str) -> List[str]:
    """Load samples from a file for reuse."""
    if sample_file and os.path.exists(sample_file):
        if sample_file.lower().endswith(".csv"):
            subset = read_csv_and_add_png(sample_file)
            print(f"Reusing {len(subset)} samples from {sample_file}")
            return subset
    
    # Fallback to candidate_paths.txt
    if os.path.exists(CANDIDATE_PATHS_FILE):
        with open(CANDIDATE_PATHS_FILE) as f:
            cand_list = [
                line.strip() if line.strip().lower().endswith(".png")
                else line.strip() + ".png"
                for line in f
            ]
        print(f"Reusing {len(cand_list)} samples from {CANDIDATE_PATHS_FILE}")
        return cand_list
    
    print("[WARN] No reusable samples found")
    return []


def load_missing_samples(
    missing_csv: str,
    image_folder: str,
    model_name: str
) -> List[str]:
    """Load samples from a missing CSV file."""
    subset = read_missing_csv_and_find_file_name(
        csv_path=missing_csv,
        image_folder_path=image_folder,
        model=model_name
    )
    print(f"Loaded {len(subset)} missing basenames from {missing_csv}")
    return subset


def load_folder_samples(
    image_folder: str,
    skip_basenames: Set[str]
) -> List[str]:
    """Load all images from a folder."""
    cand_list = list_images(image_folder)
    subset = filter_by_skip_basenames(cand_list, skip_basenames)
    print(f"Loaded {len(subset)} images from folder {image_folder}")
    return subset


def load_cityscape_default_samples(
    model_name: str,
    skip_basenames: Set[str],
    sample_size: int = DEFAULT_SAMPLE_SIZE
) -> List[str]:
    """Load default Cityscape samples from multiple cities."""
    cand_list = []
    for city in CITYSCAPE_CITIES:
        city_path = f"cityscape/defogged_models/{model_name}/val/{city}"
        cand_list.extend(list_images(city_path))
    
    filtered = filter_by_skip_basenames(cand_list, skip_basenames)
    subset = random.sample(filtered, min(sample_size, len(filtered)))
    print(f"Loaded {len(subset)} random samples from Cityscape cities")
    return subset


def load_image_subset(
    args: argparse.Namespace,
    model_name: str,
    skip_basenames: Set[str]
) -> List[str]:
    """Load the appropriate subset of images based on arguments."""
    if args.reuse_sample:
        return load_reuse_samples(args.sample_file)
    
    if args.missing_csv:
        return load_missing_samples(args.missing_csv, args.image_folder, model_name)
    
    if args.image_folder:
        return load_folder_samples(args.image_folder, skip_basenames)
    
    return load_cityscape_default_samples(model_name, skip_basenames)
