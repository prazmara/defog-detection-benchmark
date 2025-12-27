#!/usr/bin/env python3
"""
Image Defogging Pipeline using Google Gemini AI
Processes images from an input directory and saves defogged results to an output directory.
"""

import argparse
import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from tools.io import list_images


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Defog images using Google Gemini AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing foggy images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for defogged images'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Google Gemini API key (can also be set via GEMINI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash-image-preview',
        help='Gemini model to use'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='Need to remove fog so the objects should be clearly visible',
        help='Prompt for the defogging task'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=2048,
        help='Target width for output images'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=1024,
        help='Target height for output images'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for model generation'
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Filter images by substring (e.g., "0.01" to process only images with "0.01" in filename)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip images that already have output files'
    )
    
    return parser.parse_args()


def process_image(client, image_path, output_path, args):
    """
    Process a single image through the defogging pipeline.
    
    Args:
        client: Google Gemini client
        image_path: Path to input image
        output_path: Path to save defogged image
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing: {image_path}")
        
        # Load image
        image = Image.open(image_path)
        
        # Generate defogged content
        response = client.models.generate_content(
            model=args.model,
            contents=[args.prompt, image],
            config=types.GenerateContentConfig(
                temperature=args.temperature,
            )
        )
        
        # Process response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(f"  Model response: {part.text}")
                
            elif part.inline_data is not None:
                # Extract image from response
                result_image = Image.open(BytesIO(part.inline_data.data))
                
                # Resize if necessary
                target_size = (args.width, args.height)
                if result_image.size != target_size:
                    print(f"  Resizing image from {result_image.size} to {target_size}")
                    try:
                        resample = Image.Resampling.BICUBIC
                    except AttributeError:
                        resample = Image.BICUBIC
                    result_image = result_image.resize(target_size, resample)
                
                # Save result
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result_image.save(output_path)
                print(f"  Saved to: {output_path}")
                return True
                
        print(f"  Warning: No image data in response for {image_path}")
        return False
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    # Get API key
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError(
            "API key must be provided via --api-key argument or GEMINI_API_KEY environment variable"
        )
    
    # Initialize client
    print(f"Initializing Gemini client with model: {args.model}")
    client = genai.Client(api_key=api_key)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    # List images
    print(f"Scanning input directory: {args.input_dir}")
    image_list = list_images(args.input_dir)
    
    if not image_list:
        print(f"No images found in {args.input_dir}")
        return
    
    # Apply filter if specified
    if args.filter:
        image_list = [img for img in image_list if args.filter in img]
        print(f"Applied filter '{args.filter}': {len(image_list)} images match")
    
    if not image_list:
        print("No images to process after filtering")
        return
    
    print(f"Found {len(image_list)} images to process")
    
    # Process images
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, image_filename in enumerate(image_list, 1):
        print(f"\n[{idx}/{len(image_list)}] Processing: {image_filename}")
        
        # Construct paths
        input_path = os.path.join(args.input_dir, image_filename)
        
        # Generate output filename
        base, ext = os.path.splitext(image_filename)
        output_filename = f"{base}_defogged.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Check if output already exists
        if args.skip_existing and os.path.exists(output_path):
            print(f"  Skipping (output exists): {output_path}")
            skip_count += 1
            continue
        
        # Process image
        if process_image(client, input_path, output_path, args):
            success_count += 1
        else:
            error_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"  Total images: {len(image_list)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
