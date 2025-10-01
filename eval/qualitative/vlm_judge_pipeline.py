from tools.io import imread, list_images
from tools.gpt_endpoint.gpt_tool_azure import AzureVLMScorer

from tools.filename_parser import triplet_from_cand_name
import os
import dotenv
from openai import AzureOpenAI
import argparse
import os
from openai import AzureOpenAI
from tools.db.persitence import append_unique, load_existing_pairs, check_existing
import random
from tools.helpers import read_csv_and_add_png
from pathlib import Path
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=[
        "dehazeformer", "focalnet", "mitdense", "mitnh",
        "fluxnet", "nanobanana",
        "b01_dhft", "b01_dhf", "b01",
        "flux_non_cot", "flux_cot",
        "flux_split", "flux_split_cot", "flux_split_non_cot",
        "GT",
    ],
    help="Name of the model to evaluate"
)
parser.add_argument("--image-folder", default="",
                    help="Path to a folder of images to process")
parser.add_argument("--num-samples", type=int, default=50,
                    help="Number of samples to process (default: 50)")
parser.add_argument("--foggy-image-folder", default="cityscape/foggy/val",
                    help="Path to foggy images (default: cityscape/foggy/val)")
parser.add_argument("--gt-image-folder", default="cityscape/ground_truth/val",
                    help="Path to GT images (default: cityscape/ground_truth/val)")
parser.add_argument("--output-csv", default="DB/combined_results.csv",)
parser.add_argument("--skip-csv", 
                    help="Path to a CSV file with basenames to skip")
parser.add_argument("--fixed", default="fixed_basenames/candidates_paths.txt",
                    help="Path to a text file with fixed basenames to process")

args = parser.parse_args()

DEFAULT_FIELDS = [
    "model", "city", "basename",
    "foggy_path", "cand_path", "gt_path",
    "visibility_restoration", "visual_distortion", "boundary_clarity",
    "scene_consistency", "object_consistency", "perceived_detectability", "relation_consistency",
    "explanation"
]

VALID_MODELS = {
    "dehazeformer", "focalnet", "mitdense", "mitnh",
    "fluxnet", "nanobanana",
    "b01_dhft", "b01_dhf", "b01",
    "flux_non_cot", "flux_cot",
    "flux_split", "flux_split_cot", "flux_split_non_cot",
    "GT"
}


# --- env / client ---------------------------------------------------------
dotenv.load_dotenv()
subscription_key = os.getenv("AZURE_API_KEY")  # make sure this is set in your .env
api_version  = os.getenv("AZURE_API_VERSION")
azure_endpoint = os.getenv("AZURE_API_BASE")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=subscription_key,
)


scorer = AzureVLMScorer(client, deployment="gpt-5-chat", temperature=0.0)





# --- main loop ------------------------------------------------------------
if args.model:
    if args.model in VALID_MODELS:
        model_name = args.model
    else:
        raise ValueError(f"Unknown model name: {args.model}")


DB = args.output_csv


def load_skip_basenames(csv_path):
    skip_basenames = set()
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 1:
                skip_basenames.add(parts[0])
    return skip_basenames


def load_fixed_basenames(path: str) -> set[str]:
    """
    Load a set of fixed basenames from a file, skipping blanks and stripping whitespace.
    Returns an empty set if path is None or does not exist.
    """
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def check_paths(paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"  [WARN] missing file: {p}")
            return False
    return True


# 1) skip set
skip_basenames = set(load_skip_basenames(args.skip_csv)) if args.skip_csv else set()

# 2) choose source for candidates
subset = []


if args.image_folder:
    cand_list = list_images(args.image_folder)
    subset = [p for p in cand_list if Path(p).stem not in skip_basenames]
    print(f"Loaded {len(subset)} images from folder {args.image_folder}")
    
    fixed = load_fixed_basenames(args.fixed)
    if fixed:
        subset = [p for p in subset if Path(p).stem in fixed]
        print(f"Using {len(subset)} fixed basenames from {args.fixed}")
        if len(subset) == 0:
            raise ValueError("No images to process after applying fixed basenames filter")

else:
    cities = ["frankfurt", "lindau", "munster"]
    base = Path(f"cityscape/defogged_models/{model_name}/val")

    # collect candidates from all cities
    cand_list = [p for city in cities for p in list_images(base / city)]

    # apply skip filter
    cand_list = [p for p in cand_list if Path(p).stem not in skip_basenames]

    # load fixed basenames (if any)
    fixed = load_fixed_basenames(args.fixed)

    if fixed:
        subset = [p for p in cand_list if Path(p).stem in fixed]
        print(f"Using {len(subset)} fixed basenames from {args.fixed}")
    else:
        subset = random.sample(cand_list, k=min(args.num_samples, len(cand_list)))
        print(f"Sampled {len(subset)} images (num_samples={args.num_samples})")



for cand_name in subset:
    gt_image_path, foggy_image_path, cand_image_path, city = triplet_from_cand_name(
        cand_name=cand_name,
        model=model_name,
        image_folder=args.image_folder
    )

    if model_name == "GT":
        cand_image_path = gt_image_path
    elif model_name == "b01":
        cand_image_path = foggy_image_path
    
   
    
    print(f"Evaluating {cand_image_path} against {foggy_image_path} and {gt_image_path}")
    basename =cand_name.rsplit(".",1)[0] 
    if check_existing(
        csv_path=DB,
        model=model_name,
        city=city,
        basename=basename
    ):
        print(f"  [SKIP] already exists in DB {basename}")
        continue

    if not check_paths(paths=[foggy_image_path, cand_image_path, gt_image_path]):
        print("  [SKIP] missing one of the required images")
        continue


    metrics = {
        "foggy_vs_gt": {"ssim": None, "lpips": None, "black_ssim": None},
        "cand_vs_gt":  {"ssim": None, "lpips": None, "black_ssim": None},
    }


    scores = scorer.score_triplet(
        foggy_path=foggy_image_path,
        cand_path=cand_image_path,
        gt_path=gt_image_path,
        metrics=metrics,
    )
    print("VLM rubric scores:", scores)


    
    row = {
        "model": model_name,               # e.g., "dehazeformer"
        "city": city,                      # e.g., "frankfurt"
        "basename": basename,              # e.g., "frankfurt_000001_066574_leftImg8bit"
        "foggy_path": foggy_image_path,
        "cand_path": cand_image_path,
        "gt_path": gt_image_path,
        **scores
    }

    written = append_unique(
        csv_path=DB,
        row=row,
        key_fields=("model","split","city","basename"),
        fields=DEFAULT_FIELDS,
        jsonl_path="outputs/combined_results.jsonl"  # optional
    )
    if written:
        print("[WRITE] appended row")
    else:
        print("[SKIP] duplicate key")
        





