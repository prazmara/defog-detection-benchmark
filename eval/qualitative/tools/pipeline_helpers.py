# tools/pipeline_helpers.py
from __future__ import annotations
from typing import Dict
from tools.io import imread
import tools.ssim as ssim
from tools.difficulty import difficulty_vector

def ssim_color_pair(foggy_path: str, gt_path: str) -> float:
    """Return color SSIM between foggy and GT (expects file paths)."""
    fog = imread(foggy_path, mode="color")
    gt  = imread(gt_path,   mode="color")
    return float(ssim.ssim_score(fog, gt, mode="color"))

def difficulty_from_path(img_path: str, include_dark_channel: bool = True) -> Dict[str, float]:
    """Return difficulty stats for one image path."""
    img = imread(img_path, mode="color")
    return difficulty_vector(img, include_dark_channel=include_dark_channel)
