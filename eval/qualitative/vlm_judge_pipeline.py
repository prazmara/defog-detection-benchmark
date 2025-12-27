#!/usr/bin/env python3
"""
VLM Judge Pipeline - Main entry point.

Evaluates defogged images using Vision Language Models across different models
and datasets (Cityscape, ACDC).
"""

from pipeline.cli import parse_arguments, validate_model
from pipeline.azure_setup import setup_azure_clients, setup_output_directory
from pipeline.data_loader import load_skip_basenames, load_image_subset
from pipeline.image_processor import process_image_batch


def main() -> None:
    """Main entry point for the VLM judge pipeline."""
    # Parse and validate arguments
    args = parse_arguments()
    model_name = validate_model(args.model)
    
    # Setup Azure clients and output directory
    setup_output_directory(args.output_csv)
    _, scorer = setup_azure_clients()
    
    # Load data
    skip_basenames = load_skip_basenames(args.skip_csv)
    image_subset = load_image_subset(args, model_name, skip_basenames)
    
    if not image_subset:
        print("[ERROR] No images to process")
        return
    
    # Process images
    process_image_batch(
        image_subset,
        model_name,
        args.dataset,
        args.image_folder,
        scorer,
        args.output_csv
    )
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
