import os
from typing import Tuple, List

# Define templates once
CITYSCAPE_PATH_TEMPLATES = {
    "gt": "cityscape/gournd_truth/val/{city}/{city}_{seq}_{frame}_leftImg8bit{ext}",
    "foggy": "cityscape/foggy/val/{city}/{city}_{seq}_{frame}_leftImg8bit_foggy_beta_{beta}{ext}",
    "b01_dhft": "cityscape/{model}/val/{city}/{city}_{seq}_{frame}_leftImg8bit{ext}",
    "default": "cityscape/defogged_models/{model}/val/{city}/{city}_{seq}_{frame}_leftImg8bit{ext}",
}

ACDC_PATH_TEMPLATES = {
    "gt": "acdc/acdc-fog-val/{cand_name}", 
    "default": "acdc/{model}/{cand_name}"
}



def triplet_from_cand_name(
    cand_name: str,
    model: str,
    image_folder: str = None,
    fog_betas: List[float] = [0.01],
) -> Tuple[str, List[str], str, str]:
    """
    Given a candidate filename (e.g., frankfurt_000001_066574_leftImg8bit.png),
    construct the corresponding GT, foggy, and candidate paths.

    Returns:
        gt_path    : str
        foggy_paths: list of str (one per beta)
        cand_path  : str
        city       : str
    """
    base, ext = os.path.splitext(cand_name)
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected cand_name format: {cand_name}")

    city, seq, frame = parts[0], parts[1], parts[2]

    # Candidate path
    if image_folder:
        if not os.path.exists(image_folder):
            raise ValueError(f"Provided image_folder does not exist: {image_folder}")
        if image_folder == "nano_foggy_results":
            cand_path = os.path.join(image_folder, f"{city}_{seq}_{frame}_defogged_nano.png")
        else:
            cand_path = os.path.join(image_folder, cand_name)
    else:
        template_key = model if model in CITYSCAPE_PATH_TEMPLATES else "default"
        cand_path = CITYSCAPE_PATH_TEMPLATES[template_key].format(
            city=city, seq=seq, frame=frame, ext=ext, model=model
        )

    # Foggy paths (can support multiple betas now)
    foggy_paths = CITYSCAPE_PATH_TEMPLATES["foggy"].format(city=city, seq=seq, frame=frame, ext=ext, beta=str(fog_betas[0]))
    

    # Ground truth path
    gt_path = CITYSCAPE_PATH_TEMPLATES["gt"].format(city=city, seq=seq, frame=frame, ext=ext)

    return gt_path, foggy_paths, cand_path, city

def build_acdc_paths(
        model: str, 
        cand_name: str
):
    
    gt_path = ACDC_PATH_TEMPLATES["gt"].format(cand_name=cand_name)
    template_key = model if model in ACDC_PATH_TEMPLATES else "default"
    candidate_path = ACDC_PATH_TEMPLATES[template_key].format(model=model, cand_name=cand_name)
    print(f"ACDC paths: GT: {gt_path}, Candidate: {candidate_path}")
    return gt_path, candidate_path
