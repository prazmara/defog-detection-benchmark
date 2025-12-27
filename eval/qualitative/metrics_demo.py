import sys
from tools.pipeline_helpers import ssim_color_pair
from tools.lpips import compute_lpips
from tools.io import imread, imwrite


import numpy as np
import cv2
from tools.io import imread, imwrite

def match_rgb_mean_std_to_ref(src_path: str, ref_path: str, out_path: str):
    """Match per-channel mean/std of src to ref in RGB space and save as PNG."""
    src = imread(src_path, mode="color").astype(np.float32) / 255.0  # RGB expected
    ref = imread(ref_path, mode="color").astype(np.float32) / 255.0

    # compute per-channel stats
    s_mean = src.mean(axis=(0,1), keepdims=True)
    s_std  = src.std(axis=(0,1), keepdims=True) + 1e-8
    r_mean = ref.mean(axis=(0,1), keepdims=True)
    r_std  = ref.std(axis=(0,1), keepdims=True) + 1e-8

    # affine channel-wise transform
    out = (src - s_mean) / s_std * r_std + r_mean
    out = np.clip(out, 0.0, 1.0)
    imwrite(out_path, (out * 255.0).astype(np.uint8))



def imnormalize(img_path: str) -> None:

    """Normalize image to zero mean, unit variance per channel."""
    import numpy as np
    img = imread(img_path, mode="color")
    if img.dtype != np.float32:
        img = img.astype(np.float32)    
    # Convert to float in [0,1]
    img_norm = img / 255.0

    # Per-channel mean/std
    means = img_norm.mean(axis=(0,1), keepdims=True)
    stds  = img_norm.std(axis=(0,1), keepdims=True)
    img_channel_norm = (img_norm - means) / (stds + 1e-8)
    imwrite(img_path, (img_channel_norm * 255).astype(np.uint8))



def run_metrics(foggy_image_path, cand_image_path, gt_image_path):
    metrics = {
        "foggy_vs_gt": {"ssim": None, "lpips": None, "black_ssim": None},
        "cand_vs_gt":  {"ssim": None, "lpips": None, "black_ssim": None},
    }
    metrics["foggy_vs_gt"]["ssim"] = ssim_color_pair(foggy_image_path, gt_image_path)
    metrics["foggy_vs_gt"]["lpips"] = compute_lpips(foggy_image_path, gt_image_path)
    metrics["cand_vs_gt"]["ssim"]  = ssim_color_pair(cand_image_path, gt_image_path)
    metrics["cand_vs_gt"]["lpips"]  = compute_lpips(cand_image_path, gt_image_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    gt_image_path = "cityscape/gournd_truth/val/frankfurt/frankfurt_000001_082466_leftImg8bit.png"
    cand_flux_image_path = "frankfurt_000001_082466_leftImg8bit_foggy_beta_0_01_defogged_2048x1024.png"
    
    cand_dhf_image_path = "cityscape/defogged_models/dehazeformer/val/frankfurt/frankfurt_000001_082466_leftImg8bit.png"
    cand_nano_image_path = "nano_foggy_results/frankfurt_000001_082466_defogged_nano.png"
    foggy_image_path = "cityscape/foggy/val/frankfurt/frankfurt_000001_082466_leftImg8bit_foggy_beta_0.01.png"


    match_rgb_mean_std_to_ref(cand_flux_image_path, gt_image_path, "normalized_flux.png")
    print("Evaluating flux")
    run_metrics(foggy_image_path, cand_flux_image_path, gt_image_path)
    print("Evaluating dehazeformer")
    run_metrics(foggy_image_path, cand_dhf_image_path, gt_image_path)
    print("Evaluating normalized flux") 
    run_metrics(foggy_image_path, "normalized_flux.png", gt_image_path)
