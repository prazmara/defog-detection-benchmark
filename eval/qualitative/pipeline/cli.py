"""Command-line interface for VLM Judge Pipeline."""

import argparse
from pipeline.config import SUPPORTED_MODELS


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VLM Judge Pipeline for evaluating defogged images"
    )
    parser.add_argument(
        "--reuse-sample",
        action="store_true",
        help="Reuse existing sample.csv if available"
    )
    parser.add_argument(
        "--sample-file",
        default="",
        help="Path to store/load the sample basenames"
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model name. Supported: {', '.join(SUPPORTED_MODELS)}"
    )
    parser.add_argument(
        "--image-folder",
        default="",
        help="Path to a folder of images to process"
    )
    parser.add_argument(
        "--skip-csv",
        default="",
        help="CSV file with basenames to skip (optional)"
    )
    parser.add_argument(
        "--missing-csv",
        default="",
        help="CSV with missing keys/basenames to score (overrides sampling)"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV file to store results"
    )
    parser.add_argument(
        "--dataset",
        default="cityscape",
        choices=["cityscape", "acdc"],
        help="Dataset type to process"
    )
    
    return parser.parse_args()


def validate_model(model_name: str) -> str:
    """Validate the model name against supported models."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return model_name
