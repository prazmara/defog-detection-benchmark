#!/usr/bin/env python3
"""
Cityscapes Dataset Analysis with FiftyOne and YOLO

This script:
1. Loads four Cityscapes subdatasets (ground truth + 3 foggy variants) into FiftyOne
2. Visualizes the datasets with annotations
3. Performs object detection using YOLO models
4. Records and compares performance metrics
5. Evaluates model predictions against ground truth annotations
"""

import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path
import numpy as np
import time
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logfile-test.txt'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

class CityscapesAnalyzer:
    def __init__(self, data_root="./citytococo/data/cityscapes"):
        self.data_root = Path(data_root)
        self.datasets = {}
        self.models = {}
        self.results = defaultdict(dict)
        self.evaluation_results = defaultdict(dict)
        
        # Define the four subdatasets
        self.subdatasets = {
            "ground_truth": {
                "images": self.data_root / "leftImg8bit",
                "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            },
            # "foggy_beta_0.02": {
            #     "images": self.data_root / "foggy_beta_0.02",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_foggy_beta_0.02_val.json"
            # },
            "foggy_beta_0.01": {
                "images": self.data_root / "foggy_beta_0.01",
                "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            },
            # "foggy_beta_0.005": {
            #     "images": self.data_root / "foggy_beta_0.005",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_foggy_beta_0.005_val.json"
            # },
            # "dehazeformer": {
            #     "images": self.data_root / "dehazeformer",
            #     "annotations": self.data_root / "annotations" / "tak_instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "focalnet": {
            #     "images": self.data_root / "focalnet",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "mitdense": {
            #     "images": self.data_root / "mitdense",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "mitnh": {
            #     "images": self.data_root / "mitnh",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "b01_dcp": {
            #     "images": self.data_root / "foggy_beta_0.01_dcp",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "b01_msr": {
            #     "images": self.data_root / "foggy_beta_0.01_msr",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "b01_clh": {
            #     "images": self.data_root / "foggy_beta_0.01_CLAHE",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "b01_dcp_dhf": {
            #     "images": self.data_root / "b01_dcp_dhf",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "b01_dhf": {
            #     "images": self.data_root / "b01_dhf",
            #     "annotations": self.data_root / "annotations" / "instancesonly_filtered_leftImg8bit_val.json"
            # },
            # "3025": {
            #     "images": self.data_root / "b01_nano",
            #     "annotations": self.data_root / "annotations" / "not_instancesonly_filtered_leftImg8bit_val.json"
            # },
            "b01_flux_cot": {
                "images": self.data_root / "b01_flux_cot",
                "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            },
            "b01_flux_notcot": {
                "images": self.data_root / "b01_flux_notcot",
                "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            },
            "b01_fluximpg6": {
                "images": self.data_root / "b01_fluximpg6",
                "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            },
            # "tak": { 
            #     "images": self.data_root / "tak",
            #     "annotations": self.data_root / "annotations" / "tak_instancesonly_filtered_leftImg8bit_val.json"
            # }
            # "b01_ctn": {
            #     "images": self.data_root / "b01_ctn",
            #     "annotations": self.data_root / "annotations" / "tak_instancesonly_filtered_leftImg8bit_val.json"
            # }
            # "frap": {
            #     "images": self.data_root / "frap",
            #     "annotations": self.data_root / "annotations" / "frap_instancesonly_filtered_leftImg8bit_val.json"
            # }
        }
        
        # Create models directory
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        fo.config.model_zoo_dir = self.models_dir
        
    def load_coco_dataset(self, name, images_path, annotations_path):
        """Load a COCO format dataset into FiftyOne"""
        logger.info(f"Loading {name} dataset...")
        
        # Check if dataset already exists
        if name in fo.list_datasets():
            logger.info(f"Dataset {name} already exists, deleting...")
            fo.delete_dataset(name)
        
        # Load the dataset
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(images_path),
            labels_path=str(annotations_path),
            name=name,
            include_id=True
        )
        
        logger.info(f"Loaded {name}: {len(dataset)} samples")
        return dataset
    
    def load_all_datasets(self):
        """Load all four subdatasets into FiftyOne"""
        for name, paths in self.subdatasets.items():
            if paths["images"].exists() and paths["annotations"].exists():
                self.datasets[name] = self.load_coco_dataset(
                    name, paths["images"], paths["annotations"]
                )
            else:
                logger.warning(f"Missing data for {name}")
                logger.warning(f"  Images: {paths['images']} - {'exists' if paths['images'].exists() else 'missing'}")
                logger.warning(f"  Annotations: {paths['annotations']} - {'exists' if paths['annotations'].exists() else 'missing'}")
    
    def visualize_datasets(self):
        """Launch FiftyOne App to visualize all datasets"""
        logger.info("Launching FiftyOne App for visualization...")
        logger.info("Available datasets:")
        for name, dataset in self.datasets.items():
            logger.info(f"  - {name}: {len(dataset)} samples")
        
        names = list(self.datasets.keys())
        # Launch the app with the first dataset
        session = fo.launch_app(self.datasets[names[0]])
        logger.info("FiftyOne App launched. Press Ctrl+C to close and exit.")
        try:
            # Wait until user presses Ctrl+C
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Closing FiftyOne App...")
    
    def download_models(self, model_names=None):
        """Download models for object detection"""
        
        # COCO class indices for our target classes: person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7
        target_classes = [0, 1, 2, 3, 5, 6, 7]
        
        for model_name in model_names:
            # Check if model name contains "yolo" (case insensitive)
            if "yolo" in model_name.lower():
                logger.info(f"Loading YOLO model directly: {model_name}")
                try:
                    from ultralytics import YOLO
                    
                    # Extract model file name from model_name (e.g., "yolo11l-coco-torch" -> "yolo11l.pt")
                    if "yolo11" in model_name.lower():
                        if "n" in model_name.lower():
                            model_file = "yolo11n.pt"
                        elif "s" in model_name.lower():
                            model_file = "yolo11s.pt"
                        elif "m" in model_name.lower():
                            model_file = "yolo11m.pt"
                        elif "l" in model_name.lower():
                            model_file = "yolo11l.pt"
                        elif "x" in model_name.lower():
                            model_file = "yolo11x.pt"
                        else:
                            model_file = "yolo11n.pt"  # default
                    elif "yolo8" in model_name.lower():
                        if "n" in model_name.lower():
                            model_file = "yolov8n.pt"
                        elif "s" in model_name.lower():
                            model_file = "yolov8s.pt"
                        elif "m" in model_name.lower():
                            model_file = "yolov8m.pt"
                        elif "l" in model_name.lower():
                            model_file = "yolov8l.pt"
                        elif "x" in model_name.lower():
                            model_file = "yolov8x.pt"
                        else:
                            model_file = "yolov8n.pt"  # default
                    else:
                        model_file = "yolo11n.pt"  # default
                    
                    # Load YOLO model directly
                    model = YOLO(model_file)
                    
                    # Configure model with classes and other settings
                    cfg = {
                        "classes": target_classes,  # limit inference to specific classes
                    }
                    
                    for k, v in cfg.items():
                        model.overrides[k] = v
                    
                    self.models[model_name] = model
                    logger.info(f"Loaded YOLO model {model_name} with classes: {target_classes}")
                    
                except Exception as e:
                    logger.error(f"Error loading YOLO model {model_name}: {e}")
                    continue
            else:
                # Use FiftyOne zoo model for non-YOLO models
                logger.info(f"Loading FiftyOne zoo model: {model_name}")
                try:
                    foz.ensure_zoo_model_requirements(model_name)
                except Exception as e:
                    logger.error(f"Error ensuring requirements for model {model_name}: {e}")
                    continue
                
                # Load model with specific classes configuration
                try:
                    self.models[model_name] = foz.load_zoo_model(model_name)
                    logger.info(f"Loaded {model_name} ")
                except Exception as e:
                    logger.warning(f"Could not load {model_name}")
                    # Fallback to loading without classes parameter
                    self.models[model_name] = foz.load_zoo_model(model_name)
    
    def apply_detection_model(self, dataset_name, model_name, label_field):
        """Apply a YOLO model to a dataset using FiftyOne's apply_model method"""
        logger.info(f"Applying {model_name} to {dataset_name} dataset...")
        
        if model_name not in self.models:
            logger.error(f"Error: Model {model_name} not found. Please download it first.")
            return False
        
        dataset = self.datasets[dataset_name]
        model = self.models[model_name]
        
        try:
            # Apply the model to the dataset
            dataset.apply_model(model, label_field=label_field)
            logger.info(f"Successfully applied {model_name} to {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Error applying {model_name} to {dataset_name}: {e}")
            return False
    
    def apply_models_to_all_datasets(self, model_names):
        """Apply multiple models to all datasets"""
        logger.info(f"Applying models {model_names} to all datasets...")
        
        for dataset_name in self.datasets.keys():
            logger.info(f"Processing dataset: {dataset_name}")
            for model_name in model_names:
                self.apply_detection_model(dataset_name, model_name, label_field=model_name)
    
    def evaluate_detections(self, dataset_name, model_name, gt_field, eval_key):
        """Evaluate model predictions against ground truth annotations"""
        logger.info(f"Evaluating {model_name} predictions on {dataset_name} dataset...")
        
        dataset = self.datasets[dataset_name]
        
        try:
            # Evaluate detections
            detection_results = dataset.evaluate_detections(
                model_name, 
                eval_key=eval_key,
                compute_mAP=True,
                gt_field=gt_field,
                classes=["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"],
            )
            
            # Get mAP
            mAP = detection_results.mAP()
            logger.info(f"{model_name} on {dataset_name} - mAP = {mAP:.4f}")
            
            # Get class counts
            counts = dataset.count_values(f"{gt_field}.detections.label")
            
            # For Cityscapes, show all classes (we have 7 classes)
            all_classes = sorted(counts, key=counts.get, reverse=True)
            
            logger.info(f"Class-wise performance for {model_name} on {dataset_name}:")
            logger.info(detection_results.report(classes=all_classes))
            logger.info(detection_results.metrics(classes=all_classes))
            
            # Store results
            self.evaluation_results[dataset_name][model_name] = {
                "mAP": mAP,
                "detection_results": detection_results,
                "class_counts": counts
            }
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
            return None
    
    def evaluate_all_models(self, model_names, gt_field="segmentations"):
        """Evaluate all models on all datasets"""
        logger.info("Evaluating all models on all datasets...")
        
        for dataset_name in self.datasets.keys():
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            
            for model_name in model_names:
                if model_name in self.models:
                    eval_key = model_name.replace("-", "_")
                    self.evaluate_detections(dataset_name, model_name, gt_field, eval_key=eval_key)
                else:
                    logger.warning(f"Warning: Model {model_name} not available for evaluation")
    
    def print_evaluation_summary(self):
        """Print a summary of all evaluation results"""
        logger.info("EVALUATION SUMMARY")
        
        for dataset_name, model_results in self.evaluation_results.items():
            logger.info(f"Dataset: {dataset_name}")
            
            for model_name, results in model_results.items():
                mAP = results["mAP"]
                logger.info(f"{model_name:15s}: mAP = {mAP:.4f}")

def main():
    """Run complete analysis with model evaluation"""
    logger.info("="*80)
    logger.info(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Cityscapes Dataset Analysis - Evaluation Mode")
    fo.core.dataset.delete_datasets(glob_patt='*')

    
    # Initialize analyzer
    analyzer = CityscapesAnalyzer()
    
    # Step 1: Load all datasets
    logger.info("Step 1: Loading datasets...")
    analyzer.load_all_datasets()
    
    # Step 2: Download models
    logger.info("Step 2: Downloading models...")
    # yolo models 
    # RT-DTETR models
    # Faster R-CNN models
    # model_names = ["faster-rcnn-resnet50-fpn-coco-torch",
    #                "rtdetr-l-coco-torch",
    #                "rtdetr-x-coco-torch",
    #                "yolo11l-coco-torch",
    #                "yolo11n-coco-torch"]  
    model_names = ["yolo11l-coco-torch"]
    analyzer.download_models(model_names)

    # Step 3: Apply models to all datasets
    logger.info("Step 3: Applying detection models to datasets...")
    analyzer.apply_models_to_all_datasets(model_names)
    

    # Step 4: Evaluate model performance
    logger.info("Step 4: Evaluating model performance...")
    analyzer.evaluate_all_models(model_names)
    
    # Step 5: Print evaluation summary
    analyzer.print_evaluation_summary()
    
    # Step 6: Visualize predictions vs ground truth (optional)
    logger.info("Step 6: Dataset visualization with predictions")
    
    # You can uncomment the following line to launch visualization
    analyzer.visualize_datasets()


if __name__ == "__main__":
    main()
