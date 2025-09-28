import os
import glob
import json
import argparse
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from pathlib import Path
import random

CITYSCAPES_CATEGORIES = [
    {"id": 0, "name": "road", "isthing": 0, "color": [128, 64, 128]},
    {"id": 1, "name": "sidewalk", "isthing": 0, "color": [244, 35, 232]},
    {"id": 2, "name": "building", "isthing": 0, "color": [70, 70, 70]},
    {"id": 3, "name": "wall", "isthing": 0, "color": [102, 102, 156]},
    {"id": 4, "name": "fence", "isthing": 0, "color": [190, 153, 153]},
    {"id": 5, "name": "pole", "isthing": 0, "color": [153, 153, 153]},
    {"id": 6, "name": "traffic light", "isthing": 0, "color": [250, 170, 30]},
    {"id": 7, "name": "traffic sign", "isthing": 0, "color": [220, 220, 0]},
    {"id": 8, "name": "vegetation", "isthing": 0, "color": [107, 142, 35]},
    {"id": 9, "name": "terrain", "isthing": 0, "color": [152, 251, 152]},
    {"id": 10, "name": "sky", "isthing": 0, "color": [70, 130, 180]},
    {"id": 11, "name": "person", "isthing": 1, "color": [220, 20, 60]},
    {"id": 12, "name": "rider", "isthing": 1, "color": [255, 0, 0]},
    {"id": 13, "name": "car", "isthing": 1, "color": [0, 0, 142]},
    {"id": 14, "name": "truck", "isthing": 1, "color": [0, 0, 70]},
    {"id": 15, "name": "bus", "isthing": 1, "color": [0, 60, 100]},
    {"id": 16, "name": "train", "isthing": 1, "color": [0, 80, 100]},
    {"id": 17, "name": "motorcycle", "isthing": 1, "color": [0, 0, 230]},
    {"id": 18, "name": "bicycle", "isthing": 1, "color": [119, 11, 32]},
]

def id_to_color_map(segmentation):
    unique_ids = np.unique(segmentation)
    color_map = {uid: (random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
                 for uid in unique_ids}
    h, w = segmentation.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for uid, color in color_map.items():
        color_img[segmentation == uid] = color
    return color_img

def id2rgb(segmentation):
    """Convert segment id map to RGB encoding for panoptic API."""
    h, w = segmentation.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = segmentation % 256
    rgb[..., 1] = (segmentation // 256) % 256
    rgb[..., 2] = (segmentation // (256 * 256)) % 256
    return rgb

def compute_area_and_bbox(segmentation, seg_id):
    mask = (segmentation == seg_id)
    area = int(mask.sum())
    if area == 0:
        return area, [0, 0, 0, 0]
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    return area, bbox

def main(args):
    # ----------------------------
    # Load model and processor
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Mask2FormerImageProcessor.from_pretrained(args.model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()

    # ----------------------------
    # Collect images
    # ----------------------------

    img_paths = sorted(Path(args.datafolder).rglob("*_foggy_beta_0.01.png"))   
    # img_paths = sorted(Path(args.datafolder).rglob("*.png"))   
    print(f"Found {len(img_paths)}")
    print(img_paths)
    os.makedirs(args.output_dir, exist_ok=True)

    categories = [
        {"id": int(k), "name": v}
        for k, v in {
            '0': 'road', '1': 'sidewalk', '2': 'building', '3': 'wall', '4': 'fence',
            '5': 'pole', '6': 'traffic light', '7': 'traffic sign', '8': 'vegetation',
            '9': 'terrain', '10': 'sky', '11': 'person', '12': 'rider', '13': 'car',
            '14': 'truck', '15': 'bus', '16': 'train', '17': 'motorcycle', '18': 'bicycle'
        }.items()
    ]
    
    panoptic_json = {
        "annotations": [],
        "images": [],
        "categories": CITYSCAPES_CATEGORIES #model.config.id2label
    }

    # ----------------------------
    # Process in batches
    # ----------------------------
    for i in tqdm(range(0, len(img_paths), args.batch_size)):
        batch_paths = img_paths[i:i+args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1] for img in images]
        )

        # ----------------------------
        # Save results
        # ----------------------------
        for img_path, res in zip(batch_paths, results):
            file_name = os.path.basename(img_path).replace("_foggy_beta_0.01.png", ".png")
            # file_name = os.path.basename(img_path).replace(".PNG", ".png")
            seg = res["segmentation"].cpu().numpy().astype(np.uint8)
            seg_img = Image.fromarray(id2rgb(seg))
            out_path = os.path.join(args.output_dir, file_name)
            seg_img.save(out_path)

            # Save visualization (colored mask)
            color_mask = id_to_color_map(seg)
            vis_path = os.path.join(args.output_dir, file_name.replace(".png", "_vis.png"))
            Image.fromarray(color_mask).save(vis_path)

            # add to panoptic JSON
            img_id = os.path.splitext(file_name)[0]
            panoptic_json["images"].append({
                "id": img_id,
                "file_name": file_name,
                "height": seg.shape[0],
                "width": seg.shape[1],
            })
            # panoptic_json["annotations"].append({
            #     "image_id": img_id,
            #     "file_name": file_name,
            #     "segments_info": res["segments_info"]
            # })
            segments_info = []
            for seg_info in res["segments_info"]:
                seg_id = seg_info["id"]
                cat_id = int(seg_info["label_id"])  # map label_id -> category_id
                area, bbox = compute_area_and_bbox(seg, seg_id)
                segments_info.append({
                    "id": int(seg_id),
                    "category_id": cat_id,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })

            panoptic_json["annotations"].append({  
                "image_id": img_id,
                "file_name": file_name,
                "segments_info": segments_info
            })

    # ----------------------------
    # Save panoptic json
    # ----------------------------
    json_path = os.path.join(args.output_dir, "panoptic.json")
    with open(json_path, "w") as f:
        json.dump(panoptic_json, f)
    print(f"Saved results in {args.output_dir}, JSON: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mask2Former on Cityscapes dataset")
    parser.add_argument("--model_id", type=str,
                        default="facebook/mask2former-swin-large-cityscapes-panoptic",
                        help="HuggingFace model repo")
    parser.add_argument("--datafolder", type=str, required=True,
                        help="Path to city folder with *.PNG images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Where to save results")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for inference")

    args = parser.parse_args()
    main(args)
