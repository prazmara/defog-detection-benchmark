# io.py
from __future__ import annotations
import os
import pathlib
from typing import Iterable, List, Literal, Optional

import cv2
import numpy as np
from pathlib import Path
ColorMode = Literal["color", "gray"]


def ensure_dir(path: str | os.PathLike) -> None:
    """Create parent directory for a file path, if missing."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def imread(
    path: str | os.PathLike,
    mode: ColorMode = "color",
    as_float: bool = False,
) -> np.ndarray:
    """
    Read an image from disk.

    Args:
        path: image file path.
        mode: "color" -> RGB uint8 (H,W,3); "gray" -> grayscale uint8 (H,W).
        as_float: if True, return float32 in [0,1]; else uint8.

    Returns:
        np.ndarray image.

    Raises:
        FileNotFoundError if the image can't be read.
    """
    p = str(path)
    if mode == "gray":
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        if as_float:
            img = (img.astype(np.float32) / 255.0).copy()
        return img

    # color
    img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if as_float:
        img_rgb = (img_rgb.astype(np.float32) / 255.0).copy()
    return img_rgb


def imwrite(
    path: str | os.PathLike,
    img: np.ndarray,
    make_dirs: bool = True,
) -> None:
    """
    Write an image to disk. Accepts RGB, BGR, or grayscale.
    If input is RGB, converts to BGR for OpenCV.
    """
    if make_dirs:
        ensure_dir(path)

    arr = img
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Heuristic: assume RGB, convert to BGR for OpenCV.
        # If your array is already BGR, pass it directly to cv2.imwrite.
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Handle float images in [0,1]
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)

    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def list_images(
    root: str | os.PathLike,
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> List[str]:
    """
    Recursively list image files under root with given extensions.
    """
    root = pathlib.Path(root)
    exts_lower = tuple(e.lower() for e in exts)
    out: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_lower:
            out.append(str(p))
    return sorted(out)


def list_images(
    image_dir: str,
    exts={".png", ".jpg", ".jpeg"},
    keep_path: bool = False
) -> List[str]:
    """
    List image files in a flat directory.

    Args:
        image_dir: directory path
        exts: allowed extensions (case-sensitive; e.g., {".png"})
        keep_path: if True, return absolute paths; 
                   if False, return just filenames

    Returns:
        List of image names or full absolute paths
    """
    files = [
        fn for fn in os.listdir(image_dir)
        if os.path.splitext(fn)[1] in exts
    ]

    if keep_path:
        return [os.path.join(image_dir, fn) for fn in files]
    return files

def list_images_from_list(*dirs, exts=(".png", ".jpg", ".jpeg")):
    """
    Collect all images from multiple directories into one list.
    
    Args:
        *dirs: One or more directory paths (strings).
        exts:  Allowed extensions (default: png/jpg/jpeg).
    
    Returns:
        List of image file paths (absolute).
    """
    images = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"⚠️ Skipping missing dir: {d}")
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in exts:
                images.append(str(f))
    return images