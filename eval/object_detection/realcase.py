#!/usr/bin/env python3
"""
Object Detection Script using YOLOv8
Detects specific classes from images and saves results
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

# COCO class indices for our target classes: person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7
TARGET_CLASSES = [0, 1, 2, 3, 5, 6, 7]
CLASS_NAMES = {
    0: 'person',
    1: 'bicycle', 
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck'
}

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the object detector
        
        Args:
            model_path: Path to YOLO model file
        """
        self.model = YOLO(model_path)
        self.target_classes = TARGET_CLASSES
        self.class_names = CLASS_NAMES
        
    def detect_objects(self, source: str, save_dir: str = "detection_results", 
                      conf_threshold: float = 0.5, save_txt: bool = True) -> List[dict]:
        """
        Detect objects in the given source
        
        Args:
            source: Path to image or directory of images
            save_dir: Directory to save results
            conf_threshold: Confidence threshold for detections
            save_txt: Whether to save detection results as text files
            
        Returns:
            List of detection results
        """
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Run prediction with specific parameters
        results = self.model.predict(
            source=source,
            save=True,
            save_txt=save_txt,
            save_conf=True,
            conf=conf_threshold,
            classes=self.target_classes,  # Only detect target classes
            project=save_dir,
            name="detections"
        )
        
        return results
    
    def filter_detections_by_class(self, results, target_classes: List[int]) -> List[dict]:
        """
        Filter detections to only include target classes
        
        Args:
            results: YOLO detection results
            target_classes: List of class indices to keep
            
        Returns:
            Filtered detection results
        """
        filtered_results = []
        
        for result in results:
            if result.boxes is not None:
                # Get class indices and confidence scores
                class_ids = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Filter by target classes
                mask = np.isin(class_ids, target_classes)
                filtered_class_ids = class_ids[mask]
                filtered_confidences = confidences[mask]
                filtered_boxes = boxes[mask]
                
                # Create filtered result
                filtered_result = {
                    'class_ids': filtered_class_ids,
                    'confidences': filtered_confidences,
                    'boxes': filtered_boxes,
                    'class_names': [self.class_names.get(int(cls_id), f'class_{int(cls_id)}') 
                                  for cls_id in filtered_class_ids]
                }
                filtered_results.append(filtered_result)
        
        return filtered_results
    
    def visualize_detections(self, image_path: str, results: List[dict], 
                           output_path: str = None) -> np.ndarray:
        """
        Visualize detections on the image
        
        Args:
            image_path: Path to input image
            results: Detection results
            output_path: Path to save visualization (optional)
            
        Returns:
            Image with drawn detections
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Draw detections
        for result in results:
            if 'boxes' in result and len(result['boxes']) > 0:
                boxes = result['boxes']
                class_ids = result['class_ids']
                confidences = result['confidences']
                class_names = result['class_names']
                
                for i, (box, cls_id, conf, cls_name) in enumerate(zip(boxes, class_ids, confidences, class_names)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{cls_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save visualization if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to: {output_path}")
        
        return image
    
    def print_detection_summary(self, results: List[dict]):
        """
        Print a summary of detections
        
        Args:
            results: Detection results
        """
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        
        total_detections = 0
        class_counts = {}
        
        for result in results:
            if 'class_ids' in result:
                total_detections += len(result['class_ids'])
                for cls_id in result['class_ids']:
                    cls_name = self.class_names.get(int(cls_id), f'class_{int(cls_id)}')
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        print(f"Total detections: {total_detections}")
        print(f"Target classes: {[self.class_names[cls] for cls in self.target_classes]}")
        print("\nDetections by class:")
        for cls_name, count in class_counts.items():
            print(f"  {cls_name}: {count}")
        print("="*50)

def main():
    """Main function to run object detection"""
    parser = argparse.ArgumentParser(description="Object Detection with YOLOv8")
    parser.add_argument("--source", "-s", required=True, 
                       help="Path to image file or directory")
    parser.add_argument("--model", "-m", default="yolov8n.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--output", "-o", default="detection_results",
                       help="Output directory for results")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--save-txt", action="store_true",
                       help="Save detection results as text files")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Create visualization of detections")
    
    args = parser.parse_args()
    
    # Initialize detector
    print(f"Loading model: {args.model}")
    detector = ObjectDetector(args.model)
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"Error: Source path does not exist: {args.source}")
        return
    
    print(f"Processing source: {args.source}")
    print(f"Target classes: {[detector.class_names[cls] for cls in detector.target_classes]}")
    
    # Run detection
    results = detector.detect_objects(
        source=args.source,
        save_dir=args.output,
        conf_threshold=args.conf,
        save_txt=args.save_txt
    )
    
    # Filter results for target classes
    filtered_results = []
    for result in results:
        filtered = detector.filter_detections_by_class([result], detector.target_classes)
        if filtered:
            filtered_results.extend(filtered)
    
    # Print summary
    detector.print_detection_summary(filtered_results)
    
    # Create visualization if requested
    if args.visualize and os.path.isfile(args.source):
        output_path = os.path.join(args.output, "visualization.jpg")
        detector.visualize_detections(args.source, filtered_results, output_path)
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    # Example usage without command line arguments
    if len(os.sys.argv) == 1:
        # Default example
        print("Running default example...")
        print("Usage: python realcase.py --source <image_path> [options]")
        print("\nExample:")
        print("python realcase.py --source image.jpg --model yolov8n.pt --output results --visualize")
        
        # You can uncomment the following lines to run with a default image
        # detector = ObjectDetector("yolov8n.pt")
        # results = detector.detect_objects("path/to/your/image.jpg", "results")
        # detector.print_detection_summary(results)
    else:
        main()
