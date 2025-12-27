# tools/difficulty.py
from __future__ import annotations
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
from skimage.measure import shannon_entropy

# --- helpers --------------------------------------------------------------

def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    """
    Ensure grayscale uint8 image.
    Accepts RGB uint8/float in [0,1] or [0,255].
    """
    arr = img
    if arr.ndim == 3 and arr.shape[2] == 3:  # assume RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if arr.dtype.kind == "f":
        # Float: assume [0,1] or [0,255]; normalize to [0,255]
        vmax = 1.0 if arr.max() <= 1.0 else 255.0
        arr = np.clip(arr, 0.0, vmax)
        arr = (arr * (255.0 / vmax)).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr

# --- metrics --------------------------------------------------------------

def edge_density(img: np.ndarray, canny_low: int = 100, canny_high: int = 200) -> float:
    """
    Fraction of edge pixels via Canny on a grayscale version of the image.
    Returns value in [0, 1].
    """
    g = _to_gray_u8(img)
    edges = cv2.Canny(g, canny_low, canny_high)
    return float(np.count_nonzero(edges)) / float(edges.size)

def rms_contrast(img: np.ndarray) -> float:
    """
    RMS contrast on grayscale; normalized to [0,1] by dividing by 255.
    """
    g = _to_gray_u8(img).astype(np.float32)
    mu = g.mean()
    return float(np.sqrt(np.mean((g - mu) ** 2)) / 255.0)

def entropy(img: np.ndarray, base: int = 2) -> float:
    """
    Shannon entropy (skimage) on grayscale.
    Typical range ~[0,8] for uint8; higher = more complex.
    """
    g = _to_gray_u8(img)
    return float(shannon_entropy(g, base=base))/8

# --- optional: simple fog proxy (dark channel) ---------------------------

def dark_channel_proxy(img: np.ndarray, radius: int = 7) -> float:
    """
    A crude fog proxy: mean of the 'dark channel' (min over color channels,
    then min filter). Higher values can indicate heavier haze.
    Returns value in [0,1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        # make a fake 3-channel from gray
        g = _to_gray_u8(img)
        rgb = np.repeat(g[..., None], 3, axis=2)
    else:
        # ensure uint8 RGB
        if img.dtype.kind == "f":
            vmax = 1.0 if img.max() <= 1.0 else 255.0
            arr = np.clip(img, 0.0, vmax)
            rgb = (arr * (255.0 / vmax)).astype(np.uint8)
        else:
            rgb = img.astype(np.uint8)

    dark = np.min(rgb, axis=2)  # per-pixel min across channels
    ksize = 2 * radius + 1
    dark_eroded = cv2.erode(dark, np.ones((ksize, ksize), np.uint8))
    return float(dark_eroded.mean() / 255.0)

# --- combined API ---------------------------------------------------------

def difficulty_vector(
    img: np.ndarray,
    include_dark_channel: bool = False
) -> Dict[str, float]:
    """
    Compute all difficulty stats for a single image.
    Returns a dict with keys: edge_density, rms_contrast, entropy
    (+ dark_channel if requested).
    """
    out = {
        "edge_density": edge_density(img),
        "rms_contrast": rms_contrast(img),
        "entropy": entropy(img),
    }
    if include_dark_channel:
        out["dark_channel"] = dark_channel_proxy(img)
    return out
