# ssim.py
from __future__ import annotations
from typing import Literal, Tuple, Optional

import numpy as np
from skimage.metrics import structural_similarity as _ssim
import cv2  # only used for color conversion if you pass BGR by mistake

Mode = Literal["gray", "color"]


def _infer_data_range(a: np.ndarray, b: np.ndarray) -> float:
    """
    Infer an appropriate data_range for SSIM.
    - If uint8: 255
    - If float: use (max over both - min over both), fallback to 1.0
    """
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        return 255.0
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    rng = hi - lo
    return rng if rng > 0 else 1.0


def ssim_gray(img1: np.ndarray, img2: np.ndarray, return_map: bool = False) -> Tuple[float, Optional[np.ndarray]]:
    if img1.ndim == 3 and img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2.ndim == 3 and img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    data_range = _infer_data_range(img1, img2)
    if return_map:
        score, s_map = _ssim(img1, img2, data_range=data_range, full=True)
        return float(score), s_map
    else:
        score = _ssim(img1, img2, data_range=data_range, full=False)
        return float(score), None



def ssim_color(img1: np.ndarray, img2: np.ndarray, return_map: bool = False) -> Tuple[float, Optional[np.ndarray]]:
    if img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError("ssim_color expects RGB images with shape (H,W,3).")
    if img2.ndim != 3 or img2.shape[2] != 3:
        raise ValueError("ssim_color expects RGB images with shape (H,W,3).")
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")

    data_range = _infer_data_range(img1, img2)
    if return_map:
        score, s_map = _ssim(img1, img2, channel_axis=-1, data_range=data_range, full=True)
        return float(score), s_map
    else:
        score = _ssim(img1, img2, channel_axis=-1, data_range=data_range, full=False)
        return float(score), None


def ssim_score(img1: np.ndarray, img2: np.ndarray, mode: Mode = "gray") -> float:
    """
    Convenience wrapper:
      - mode="gray": convert to grayscale and compute SSIM
      - mode="color": use multichannel SSIM (expects RGB)
    """
    if mode == "gray":
        s, _ = ssim_gray(img1, img2, return_map=False)
        return s
    elif mode == "color":
        s, _ = ssim_color(img1, img2, return_map=False)
        return s
    else:
        raise ValueError(f"Unknown mode: {mode}")
