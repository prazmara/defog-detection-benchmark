#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Set

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image


COT_PROMPT = (
    "Remove fog step by step, mist, and atmospheric haze. Produce a crystal-clear image with sharp object boundaries "
    "and high local contrast. Ensure cars, pedestrians, traffic signs, and buildings are distinctly visible "
    "with accurate shapes, edges, and textures. Preserve natural lighting and realistic colors. "
    "Photo-realistic; optimized for object detection."
)

NEGATIVE_PROMPT = (
    "fog, haze, mist, blur, soft edges, low contrast, washed out colors, glow, bloom, "
    "overexposure, underexposure, unrealistic textures, oversaturation, artistic stylization, "
    "distortion, hallucinated objects, artifacts, noise"
)

NOTCOT_PROMPT = "Need to remove fog"


def _read_basenames_from_csv(csv_path: Path) -> List[str]:
    basenames: List[str] = []
    if not csv_path.exists():
        return basenames
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # First row might be a header like "basename"
            cell = row[0].strip()
            if cell.lower() == "basename":
                continue
            basenames.append(cell)
    return basenames


def _write_basenames_to_csv(csv_path: Path, basenames: Iterable[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["basename"])
        for b in basenames:
            writer.writerow([b])


def _collect_paths_from_basenames(folder: Path, basenames: Iterable[str], ext: str) -> List[Path]:
    paths: List[Path] = []
    for b in basenames:
        # Try exact match first
        candidate = folder / f"{b}{ext}"
        if candidate.exists():
            paths.append(candidate)
            continue
        # Fallback: any file starting with the basename (helps if there are suffixes)
        matches = sorted(folder.glob(f"{b}*{ext}"))
        if matches:
            paths.append(matches[0])
    return paths


def gather_image_list(
    image_folder: Path,
    glob_pattern: str,
    sample_file: Optional[Path],
    reuse_sample: bool,
    skip_csv: Optional[Path],
    resume_after: Optional[str],
) -> List[Path]:
    """Return a sorted list of image Paths to process."""
    image_folder = image_folder.resolve()
    skip_set: Set[str] = set(_read_basenames_from_csv(skip_csv) if skip_csv else [])

    # If reuse_sample and sample_file exists, read basenames from it.
    if reuse_sample and sample_file and sample_file.exists():
        basenames = [b for b in _read_basenames_from_csv(sample_file) if b and b not in skip_set]
        paths = _collect_paths_from_basenames(image_folder, basenames, ext=".png")
    else:
        paths = sorted(image_folder.glob(glob_pattern))
        # Save a sample file if requested
        if sample_file:
            _write_basenames_to_csv(sample_file, (p.stem for p in paths))

        # Drop any paths that are in the skip list
        paths = [p for p in paths if p.stem not in skip_set]

    if resume_after:
        # Keep files *after* the provided basename
        try:
            idx = next(i for i, p in enumerate(paths) if p.name == resume_after or p.stem == Path(resume_after).stem)
            paths = paths[idx + 1 :]
        except StopIteration:
            # If the marker isn't found, process all files.
            pass

    return paths


def build_pipeline(device: str = "cuda", dtype: str = "bfloat16") -> FluxKontextPipeline:
    torch_dtype = torch.bfloat16 if dtype.lower() in ["bfloat16", "bf16"] else torch.float16
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch_dtype,
    ).to(device)
    return pipe


def call_pipe(
    pipe: FluxKontextPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    generator = torch.Generator(device=str(pipe.device)).manual_seed(seed)
    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    return result


def defog_resize_mode(
    pipe: FluxKontextPipeline,
    img: Image.Image,
    cot: bool,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    w, h = img.size
    prompt = COT_PROMPT if cot else NOTCOT_PROMPT
    negative = NEGATIVE_PROMPT if cot else None
    out = call_pipe(pipe, img, prompt, negative, w, h, guidance_scale, steps, seed)

    # Ensure saved output matches original size exactly
    if out.size != (w, h):
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        out = out.resize((w, h), resample)
    return out


def defog_split_mode(
    pipe: FluxKontextPipeline,
    img: Image.Image,
    cot: bool,
    guidance_scale: float,
    steps: int,
    seed: int,
) -> Image.Image:
    w, h = img.size
    if (w, h) != (2048, 1024):
        raise ValueError(f"Split mode expects 2048x1024 images, got {w}x{h}")

    prompt = COT_PROMPT if cot else NOTCOT_PROMPT
    negative = NEGATIVE_PROMPT if cot else None

    left = img.crop((0, 0, 1024, 1024)).convert("RGB")
    right = img.crop((1024, 0, 2048, 1024)).convert("RGB")

    left_out = call_pipe(pipe, left, prompt, negative, 1024, 1024, guidance_scale, steps, seed)
    right_out = call_pipe(pipe, right, prompt, negative, 1024, 1024, guidance_scale, steps, seed)

    # Concatenate back
    concat = Image.new("RGB", (2048, 1024))
    concat.paste(left_out, (0, 0))
    concat.paste(right_out, (1024, 0))
    return concat


def main():
    p = argparse.ArgumentParser(description="Unified Flux defogging driver (resize vs split, CoT vs not-CoT).")
    p.add_argument("--image-folder", required=True, help="Path to the folder of input images")
    p.add_argument("--out-folder", required=True, help="Where to save defogged results")
    p.add_argument("--mode", choices=["resize", "split"], default="resize", help="Processing mode")
    p.add_argument("--cot", dest="cot", action="store_true", help="Use Chain-of-Thought style prompt + negative prompt")
    p.add_argument("--no-cot", dest="cot", action="store_false", help="Use minimal prompt (no negative prompt)")
    p.set_defaults(cot=True)

    # Misc configuration
    p.add_argument("--glob", default="*0.01.png", help="Glob to select images (default: '*0.01.png')")
    p.add_argument("--reuse-sample", action="store_true", help="Reuse existing sample.csv if available")
    p.add_argument("--sample-file", default="", help="Path to store/load the sample basenames (CSV)")
    p.add_argument("--skip-csv", default="", help="CSV file with basenames to skip (optional)")
    p.add_argument("--resume-after", default="", help="File name or stem; only process files AFTER this one")
    p.add_argument("--model", default="flux", help="Model name (only 'flux' supported here)")

    # Inference parameters
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda", help="Device for inference, e.g., 'cuda' or 'cpu'")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="Torch dtype for the pipeline")

    args = p.parse_args()

    image_folder = Path(args.image_folder)
    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    sample_path = Path(args.sample_file) if args.sample_file else None
    skip_path = Path(args.skip_csv) if args.skip_csv else None
    resume_marker = args.resume_after or None

    paths = gather_image_list(
        image_folder=image_folder,
        glob_pattern=args.glob,
        sample_file=sample_path,
        reuse_sample=args.reuse_sample,
        skip_csv=skip_path,
        resume_after=resume_marker,
    )

    print(f"Found {len(paths)} file(s). Mode={args.mode}, CoT={args.cot}")
    pipe = build_pipeline(device=args.device, dtype=args.dtype)

    for pth in paths:
        try:
            img = load_image(str(pth)).convert("RGB")

            if args.mode == "resize":
                out = defog_resize_mode(pipe, img, args.cot, args.guidance_scale, args.steps, args.seed)
            else:
                out = defog_split_mode(pipe, img, args.cot, args.guidance_scale, args.steps, args.seed)

            # Save with _defogged suffix
            out_path = out_folder / f"{pth.stem}_defogged.png"
            out.save(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed on {pth.name}: {e}")


if __name__ == "__main__":
    main()
