import cv2
import numpy as np

from pathlib import Path

def add_suffix_to_png(path: str, suffix: str = "_X") -> str:
    p = Path(path)
    if p.suffix.lower() == ".png":
        return str(p.with_name(p.stem + suffix + p.suffix))
    return str(p)  # uncha

# -----------------------------
# Utilities
# -----------------------------
########################################## MSR
def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def _to_uint8(img):
    img = np.clip(img, 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _gaussian_blur_channel(channel, sigma):
    # Kernel size as function of sigma (odd, >=3)
    k = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(channel, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)

def _log_safe(x, eps=1e-6):
    return np.log(np.maximum(x, eps))

def _gray_world_white_balance(img):
    # Simple gray-world white balance on float image [0,1]
    mean = img.reshape(-1, 3).mean(axis=0)
    scale = mean.mean() / (mean + 1e-6)
    return np.clip(img * scale, 0, 1)

def _simple_color_balance(img, s1=1.0, s2=1.0):
    """
    Percentile clipping per channel: s1 lower %, s2 upper %
    """
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[..., c]
        lo, hi = np.percentile(ch, [s1, 100 - s2])
        if hi - lo < 1e-6:
            out[..., c] = ch
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out

# -----------------------------
# Retinex variants
# -----------------------------
def ssr(img_bgr, sigma=80):
    """
    Single-Scale Retinex (SSR) on BGR image.
    sigma: Gaussian sigma for illumination estimation (larger -> stronger global light removal).
    """
    img = _to_float(img_bgr)
    out = np.zeros_like(img)

    for c in range(3):
        I = img[..., c]
        L = _gaussian_blur_channel(I, sigma)      # illumination
        R = _log_safe(I) - _log_safe(L)           # reflectance (log domain)
        # Normalize to [0,1] per channel
        R -= R.min()
        denom = (R.max() - R.min())
        if denom < 1e-6:
            out[..., c] = I
        else:
            out[..., c] = R / denom

    # optional post WB + color balance (light touch)
    out = _gray_world_white_balance(out)
    out = _simple_color_balance(out, 1.0, 1.0)
    return _to_uint8(out)

def msr(img_bgr, sigmas=(15, 80, 250), weights=None):
    """
    Multi-Scale Retinex (MSR): weighted sum of SSR across multiple sigmas.
    sigmas: tuple/list of Gaussian sigmas.
    weights: same length as sigmas, defaults to uniform.
    """
    img = _to_float(img_bgr)
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    out = np.zeros_like(img)
    for c in range(3):
        I = img[..., c]
        R_sum = 0.0
        for w, sigma in zip(weights, sigmas):
            L = _gaussian_blur_channel(I, sigma)
            R = _log_safe(I) - _log_safe(L)
            R_sum += w * R

        # Normalize to [0,1]
        R_sum -= R_sum.min()
        denom = (R_sum.max() - R_sum.min())
        out[..., c] = R_sum / denom if denom >= 1e-6 else I

    out = _gray_world_white_balance(out)
    out = _simple_color_balance(out, 1.0, 1.0)
    return _to_uint8(out)

def msrcr(
    img_bgr,
    sigmas=(15, 80, 250),
    weights=None,
    alpha=125.0,
    beta=46.0,
    gain=1.0,
    offset=0.0,
    color_balance=(1.0, 1.0)
):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR).
    - alpha, beta: parameters for color restoration function.
    - gain, offset: linear stretch after restoration.
    - color_balance: (low%, high%) percentile clip per channel at the very end (e.g., (1,1)).
    """
    img = _to_float(img_bgr)
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)

    # MSR (log domain)
    R = np.zeros_like(img)
    for c in range(3):
        I = img[..., c]
        R_c = 0.0
        for w, sigma in zip(weights, sigmas):
            L = _gaussian_blur_channel(I, sigma)
            R_c += w * (_log_safe(I) - _log_safe(L))
        R[..., c] = R_c

    # Color restoration term (per Jobson et al.)
    sum_I = img.sum(axis=2, keepdims=True) + 1e-6
    C = beta * (_log_safe(alpha * img + 1.0) - _log_safe(sum_I))

    # MSRCR
    MSRCR = gain * (R * C) + offset

    # Normalize to [0,1] globally & mild tone mapping
    MSRCR -= MSRCR.min()
    denom = MSRCR.max() - MSRCR.min()
    if denom >= 1e-6:
        MSRCR = MSRCR / denom
    else:
        MSRCR = img

    MSRCR = _simple_color_balance(MSRCR, color_balance[0], color_balance[1])
    return _to_uint8(MSRCR)

# -----------------------------
# Example usage
# -----------------------------


    # Quick visual check (press any key to close)

def msr_filter(path):
    bgr = cv2.imread(path)
    msr_out  = msr(bgr, sigmas=(15, 80, 250))
    return msr_out

######################################### CLAHE

import cv2
import numpy as np

def CLAHE_filter(path):
  # Read the input image
  img = cv2.imread(path)

  # Convert to LAB color space
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

  # Split into channels
  l, a, b = cv2.split(lab)

  # Apply CLAHE to the L-channel (luminance)
  clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
  cl = clahe.apply(l)

  # Merge channels back
  limg = cv2.merge((cl, a, b))

  # Convert back to BGR color space
  final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

  # Show results
  # cv2.imshow("Original", img)
  # cv2.imshow("CLAHE", final)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  return final

################################################ DCP
import cv2
import numpy as np

def dcp_dehaze_file(input_path, output_path = None):
    """
    Dark Channel Prior dehazing with guided filter refinement.
    Reads a color image from `input_path`, writes the dehazed image to `output_path`.
    Keeps output size identical to input.

    Tunables (edit inside the function if needed):
        patch       = 7       # half-window radius for dark channel
        omega       = 0.95    # haze removal strength
        t0          = 0.10    # lower bound for transmission
        gf_radius   = 40      # guided filter radius
        gf_eps      = 1e-3    # guided filter epsilon
    """
    # --- Tunables ---
    patch     = 7
    omega     = 0.95
    t0        = 0.10
    gf_radius = 40
    gf_eps    = 1e-3

    # ---------------------------
    # Nested helpers (no imports)
    # ---------------------------
    def im2float(img):
        return img.astype(np.float32) / 255.0

    def dark_channel(img, radius=7):
        # img is float32 RGB [0,1]
        min_per_pixel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
        return cv2.erode(min_per_pixel, kernel, borderType=cv2.BORDER_REPLICATE)

    def estimate_atmospheric_light(img, dark, top_percent=0.001):
        h, w = dark.shape
        num_pixels = h * w
        num_top = max(1, int(num_pixels * top_percent))
        flat_dark = dark.reshape(-1)
        indices = np.argpartition(flat_dark, -num_top)[-num_top:]
        flat_img = img.reshape(-1, 3)
        candidate_pixels = flat_img[indices]
        intensities = candidate_pixels.sum(axis=1)
        return candidate_pixels[np.argmax(intensities)]

    def boxfilter(img, r):
        ksize = 2 * r + 1
        return cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize), borderType=cv2.BORDER_REFLECT)

    def guided_filter(I, p, r=40, eps=1e-3):
        I = I.astype(np.float32)
        p = p.astype(np.float32)
        ones = np.ones_like(I, dtype=np.float32)
        N = boxfilter(ones, r)
        mean_I  = boxfilter(I, r) / N
        mean_p  = boxfilter(p, r) / N
        mean_Ip = boxfilter(I * p, r) / N
        cov_Ip  = mean_Ip - mean_I * mean_p
        mean_II = boxfilter(I * I, r) / N
        var_I   = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = boxfilter(a, r) / N
        mean_b = boxfilter(b, r) / N
        return mean_a * I + mean_b

    def estimate_transmission(img, A, omega=0.95, radius=7):
        normed = img / (A.reshape(1, 1, 3) + 1e-8)
        raw_dark = dark_channel(normed, radius=radius)
        t = 1.0 - omega * raw_dark
        return np.clip(t, 0.0, 1.0)

    def recover_radiance(img, t, A, t0=0.1):
        t3 = np.maximum(t, t0)[:, :, None]
        J = (img - A.reshape(1, 1, 3)) / t3 + A.reshape(1, 1, 3)
        return np.clip(J, 0.0, 1.0)

    def enforce_same_size(arr, target_hw, name):
        th, tw = target_hw
        if arr.ndim == 2:
            h, w = arr.shape
            if (h, w) == (th, tw):
                return arr
            return cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            h, w, c = arr.shape
            if (h, w) == (th, tw):
                return arr
            out = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)
            if c == 3 and out.ndim == 2:
                out = out[:, :, None].repeat(3, axis=2)
            return out

    # ---------------------------
    # Load
    # ---------------------------
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    H, W = bgr.shape[:2]

    # ---------------------------
    # DCP pipeline (same size)
    # ---------------------------
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    I = im2float(rgb)

    dark = dark_channel(I, radius=patch)
    dark = enforce_same_size(dark, (H, W), "dark_channel")

    A = estimate_atmospheric_light(I, dark, top_percent=0.001)

    t_raw = estimate_transmission(I, A, omega=omega, radius=patch)
    t_raw = enforce_same_size(t_raw, (H, W), "transmission_raw")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = enforce_same_size(gray, (H, W), "gray_guidance")
    t_refined = guided_filter(gray, t_raw, r=gf_radius, eps=gf_eps)
    t_refined = enforce_same_size(t_refined, (H, W), "transmission_refined")
    t_refined = np.clip(t_refined, 0.0, 1.0)

    J = recover_radiance(I, t_refined, A, t0=t0)
    J = enforce_same_size(J, (H, W), "recovered_radiance")

    out_rgb_u8 = (J * 255.0 + 0.5).astype(np.uint8)
    out_bgr_u8 = cv2.cvtColor(out_rgb_u8, cv2.COLOR_RGB2BGR)

    # ---------------------------
    # Return processed image
    # ---------------------------
    return out_bgr_u8


# -----------------------------
# Script to apply filters to foggy cityscapes
# -----------------------------

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    def create_output_directories(base_path, beta_value):
        """Create output directories with the same structure as input."""
        # Create three output directories for each filter
        for filter_name in ['msr', 'CLAHE', 'dcp']:
            output_dir = base_path.parent / f"foggy_beta_{beta_value}_{filter_name}"
            output_dir.mkdir(exist_ok=True)
            
            # Create the val subdirectory
            val_dir = output_dir / "val"
            val_dir.mkdir(exist_ok=True)
            
            # Create city subdirectories
            for city in ['frankfurt', 'lindau', 'munster']:
                city_dir = val_dir / city
                city_dir.mkdir(exist_ok=True)
    
    def apply_filters_to_image(input_path, beta_value):
        """Apply all three filters to a single image and save to appropriate output folders."""
        try:
            # Get the relative path from the base foggy folder
            rel_path = input_path.relative_to(Path(f"citytococo/data/cityscapes/foggy_beta_{beta_value}"))
            
            # Apply MSR filter
            msr_out = msr_filter(str(input_path))
            msr_output_path = Path(f"citytococo/data/cityscapes/foggy_beta_{beta_value}_msr") / rel_path
            msr_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(msr_output_path), msr_out)
            
            # Apply CLAHE filter
            clahe_out = CLAHE_filter(str(input_path))
            clahe_output_path = Path(f"citytococo/data/cityscapes/foggy_beta_{beta_value}_CLAHE") / rel_path
            clahe_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clahe_output_path), clahe_out)
            
            # Apply DCP filter
            dcp_out = dcp_dehaze_file(str(input_path))
            dcp_output_path = Path(f"citytococo/data/cityscapes/foggy_beta_{beta_value}_dcp") / rel_path
            dcp_output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dcp_output_path), dcp_out)
            
            print(f"✓ Applied all filters to: {input_path.name}")
            
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}")
    
    def process_folder(beta_value):
        """Process all images in a foggy_beta folder."""
        base_path = Path(f"citytococo/data/cityscapes/foggy_beta_{beta_value}")
        
        if not base_path.exists():
            print(f"Error: {base_path} does not exist!")
            return
        
        print(f"\nProcessing foggy_beta_{beta_value}...")
        
        # Create output directories
        create_output_directories(base_path, beta_value)
        
        # Find all PNG images
        image_files = list(base_path.rglob("*.png"))
        total_images = len(image_files)
        
        print(f"Found {total_images} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {i}/{total_images}: {image_path.name}")
            apply_filters_to_image(image_path, beta_value)
        
        print(f"Completed processing foggy_beta_{beta_value}")
    
    def main():
        """Main function to process both foggy folders."""
        print("Starting filter application to foggy cityscapes images...")
        
        # Process both beta values
        beta_values = ['0.01', '0.02']
        
        for beta in beta_values:
            process_folder(beta)
        
        print("\nAll filters applied successfully!")
        print("\nOutput folders created:")
        for beta in beta_values:
            for filter_name in ['msr', 'CLAHE', 'dcp']:
                print(f"  - foggy_beta_{beta}_{filter_name}/")
    
    # Run the main function
    main()


