"""Image processing and evaluation for VLM Judge Pipeline."""

import os
from typing import Dict, List, Tuple

from tools.gpt_endpoint.gpt_tool_azure import AzureVLMScorer
from tools.filename_parser import triplet_from_cand_name, build_acdc_paths
from tools.db.persitence import append_unique, check_existing
from pipeline.config import DEFAULT_FIELDS, JSONL_OUTPUT_PATH


def check_paths(paths: List[str]) -> bool:
    """Check if all paths exist."""
    for p in paths:
        if not os.path.exists(p):
            print(f"  [WARN] missing file: {p}")
            return False
    return True


def get_image_paths_acdc(
    model_name: str,
    cand_name: str
) -> Tuple[str, str]:
    """Get image paths for ACDC dataset."""
    gt_image_path, cand_image_path = build_acdc_paths(
        model=model_name,
        cand_name=cand_name
    )
    print(f"Evaluating {cand_image_path} against {gt_image_path}")
    return gt_image_path, cand_image_path


def get_image_paths_cityscape(
    model_name: str,
    cand_name: str,
    image_folder: str
) -> Tuple[str, str, str, str]:
    """Get image paths for Cityscape dataset."""
    gt_image_path, foggy_image_path, cand_image_path, city = triplet_from_cand_name(
        cand_name=cand_name,
        model=model_name,
        image_folder=image_folder
    )
    
    # Handle special models
    if model_name == "GT":
        cand_image_path = gt_image_path
    elif model_name == "b01":
        cand_image_path = foggy_image_path
    
    print(f"Evaluating {cand_image_path} against {foggy_image_path} and {gt_image_path}")
    return gt_image_path, foggy_image_path, cand_image_path, city


def create_metrics_dict() -> Dict[str, Dict[str, None]]:
    """Create empty metrics dictionary."""
    return {
        "foggy_vs_gt": {"ssim": None, "lpips": None, "black_ssim": None},
        "cand_vs_gt": {"ssim": None, "lpips": None, "black_ssim": None},
    }


def score_image_acdc(
    scorer: AzureVLMScorer,
    cand_image_path: str,
    gt_image_path: str
) -> Dict:
    """Score images using ACDC evaluation."""
    metrics = create_metrics_dict()
    scores = scorer.score_double(
        cand_path=cand_image_path,
        gt_path=gt_image_path,
        metrics=metrics,
    )
    return scores


def score_image_cityscape(
    scorer: AzureVLMScorer,
    foggy_image_path: str,
    cand_image_path: str,
    gt_image_path: str
) -> Dict:
    """Score images using Cityscape evaluation."""
    metrics = create_metrics_dict()
    scores = scorer.score_triplet(
        foggy_path=foggy_image_path,
        cand_path=cand_image_path,
        gt_path=gt_image_path,
        metrics=metrics,
    )
    return scores


def create_result_row(
    model_name: str,
    city: str,
    basename: str,
    foggy_image_path: str,
    cand_image_path: str,
    gt_image_path: str,
    scores: Dict
) -> Dict:
    """Create a result row dictionary."""
    return {
        "model": model_name,
        "city": city,
        "basename": basename,
        "foggy_path": foggy_image_path,
        "cand_path": cand_image_path,
        "gt_path": gt_image_path,
        **scores
    }


def process_single_image(
    cand_name: str,
    model_name: str,
    dataset: str,
    image_folder: str,
    scorer: AzureVLMScorer,
    output_csv: str
) -> bool:
    """Process a single image and save results."""
    basename = cand_name.rsplit(".", 1)[0]
    
    # Get image paths based on dataset
    if dataset == "acdc":
        gt_image_path, cand_image_path = get_image_paths_acdc(model_name, cand_name)
        foggy_image_path = ""  # Not used in ACDC
        city = ""  # Not used in ACDC
        paths = [gt_image_path, cand_image_path]
    elif dataset == "cityscape":
        gt_image_path, foggy_image_path, cand_image_path, city = get_image_paths_cityscape(
            model_name, cand_name, image_folder
        )
        paths = [foggy_image_path, cand_image_path, gt_image_path]
    else:
        print(f"  [ERROR] Unknown dataset: {dataset}")
        return False
    
    # Check if already processed
    if check_existing(
        csv_path=output_csv,
        model=model_name,
        city=city,
        basename=basename
    ):
        print(f"  [SKIP] already exists in DB: {basename}")
        return False
    
    # Check if all paths exist
    if not check_paths(paths):
        print("  [SKIP] missing one of the required images")
        return False
    
    # Score the image
    if dataset == "acdc":
        scores = score_image_acdc(scorer, cand_image_path, gt_image_path)
    else:  # cityscape
        scores = score_image_cityscape(scorer, foggy_image_path, cand_image_path, gt_image_path)
    
    print(f"VLM rubric scores: {scores}")
    
    # Create and save result row
    row = create_result_row(
        model_name, city, basename,
        foggy_image_path, cand_image_path, gt_image_path,
        scores
    )
    
    written = append_unique(
        csv_path=output_csv,
        row=row,
        key_fields=("model", "split", "city", "basename"),
        fields=DEFAULT_FIELDS,
        jsonl_path=JSONL_OUTPUT_PATH
    )
    
    if written:
        print("[WRITE] appended row")
    else:
        print("[SKIP] duplicate key")
    
    return written


def process_image_batch(
    image_subset: List[str],
    model_name: str,
    dataset: str,
    image_folder: str,
    scorer: AzureVLMScorer,
    output_csv: str
) -> None:
    """Process a batch of images."""
    total = len(image_subset)
    processed = 0
    skipped = 0
    
    print(f"\nProcessing {total} images...")
    print("=" * 80)
    
    for idx, cand_name in enumerate(image_subset, 1):
        print(f"\n[{idx}/{total}] Processing: {cand_name}")
        
        success = process_single_image(
            cand_name, model_name, dataset, image_folder,
            scorer, output_csv
        )
        
        if success:
            processed += 1
        else:
            skipped += 1
    
    print("\n" + "=" * 80)
    print(f"Completed: {processed} processed, {skipped} skipped, {total} total")
