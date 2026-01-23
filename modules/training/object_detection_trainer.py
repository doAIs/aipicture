"""
Object Detection Trainer - Fine-tune object detection models (YOLO)
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, Callable
from pathlib import Path
import yaml
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class ObjectDetectionTrainer:
    """
    Trainer for object detection models using YOLO
    Supports YOLOv5, YOLOv8 and other YOLO variants
    """
    
    def __init__(self, base_model: str = "yolov8n.pt", 
                 dataset_path: str = None, output_dir: str = "./object_detection_output"):
        """
        Initialize the object detection trainer
        
        Args:
            base_model: Pre-trained model name or path
            dataset_path: Path to the dataset directory
            output_dir: Output directory for trained model
        """
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if ultralytics is available
        try:
            import ultralytics
            self.ultralytics_available = True
        except ImportError:
            self.ultralytics_available = False
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    
    def create_yaml_config(self, dataset_path: str, output_path: str = None):
        """
        Create a YAML configuration file for YOLO training
        
        Args:
            dataset_path: Path to the dataset directory
            output_path: Path for the YAML config file (optional)
        
        Returns:
            Path to the created YAML config file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "dataset.yaml")
        
        # Parse the dataset structure to find classes
        labels_dir = os.path.join(dataset_path, "labels")
        if os.path.exists(labels_dir):
            # Look for class names in the labels
            all_labels = set()
            for split in ["train", "val", "test"]:
                split_labels_dir = os.path.join(labels_dir, split)
                if os.path.exists(split_labels_dir):
                    for file in os.listdir(split_labels_dir):
                        if file.endswith('.txt'):
                            with open(os.path.join(split_labels_dir, file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        all_labels.add(int(parts[0]))  # Class ID
            
            # Create class names based on found IDs (simple mapping)
            names = [f"class_{i}" for i in sorted(all_labels)] if all_labels else ["object"]
        else:
            # Default to one class if labels not found
            names = ["object"]
        
        # Create YAML config
        config = {
            'path': dataset_path,  # dataset root dir
            'train': 'images/train',  # train images (relative to 'path')
            'val': 'images/val',      # val images (relative to 'path')
            'test': 'images/test',    # test images (optional)
            'nc': len(names),         # number of classes
            'names': names            # class names
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return output_path
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate the dataset structure for YOLO training
        
        Args:
            dataset_path: Path to the dataset directory
        
        Returns:
            True if dataset is valid, False otherwise
        """
        required_dirs = [
            os.path.join(dataset_path, "images", "train"),
            os.path.join(dataset_path, "images", "val"),
            os.path.join(dataset_path, "labels", "train"),
            os.path.join(dataset_path, "labels", "val")
        ]
        
        for req_dir in required_dirs:
            if not os.path.exists(req_dir):
                print(f"Missing required directory: {req_dir}")
                return False
        
        return True
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              learning_rate: float = 0.01,
              momentum: float = 0.937,
              weight_decay: float = 0.0005,
              warmup_epochs: int = 3.0,
              warmup_momentum: float = 0.8,
              warmup_bias_lr: float = 0.1,
              box_loss_gain: float = 0.05,
              cls_loss_gain: float = 0.5,
              cls_pw_loss_gain: float = 1.0,
              obj_loss_gain: float = 1.0,
              obj_pw_loss_gain: float = 1.0,
              iou_threshold: float = 0.2,
              anchor_threshold: float = 4.0,
              freeze_layers: int = 0,
              patience: int = 100,
              progress_callback: Optional[Callable] = None):
        """
        Train the object detection model using YOLO
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
            learning_rate: Learning rate
            momentum: Momentum for optimizer
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            warmup_momentum: Warmup momentum
            warmup_bias_lr: Warmup bias learning rate
            box_loss_gain: Box loss gain
            cls_loss_gain: Classification loss gain
            cls_pw_loss_gain: Classification positive weight loss gain
            obj_loss_gain: Objectness loss gain
            obj_pw_loss_gain: Objectness positive weight loss gain
            iou_threshold: IOU threshold
            anchor_threshold: Anchor threshold
            freeze_layers: Number of layers to freeze
            patience: Patience for early stopping
            progress_callback: Callback for progress updates
        
        Returns:
            Training result
        """
        if not self.ultralytics_available:
            raise ImportError("ultralytics is required for YOLO training. Install with: pip install ultralytics")
        
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        # Validate dataset
        if not self.validate_dataset(self.dataset_path):
            raise ValueError("Invalid dataset structure. Check that required directories exist.")
        
        # Create YAML config
        yaml_config_path = self.create_yaml_config(self.dataset_path)
        
        # Import YOLO
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(self.base_model)
        
        # Train the model
        print(f"Starting YOLO training for {epochs} epochs...")
        
        results = model.train(
            data=yaml_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,  # Initial learning rate
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box_loss_gain,
            cls=cls_loss_gain,
            dfl=cls_pw_loss_gain,
            obj=obj_loss_gain,
            obj_pw=obj_pw_loss_gain,
            iou=iou_threshold,
            anchor_t=anchor_threshold,
            freeze=freeze_layers,
            patience=patience,
            project=self.output_dir,
            name="train",
            exist_ok=True
        )
        
        # Save the model
        model.save(os.path.join(self.output_dir, "best.pt"))
        
        # Prepare result
        final_map50 = results.metrics.map50 if hasattr(results, 'metrics') and results.metrics else 0.0
        final_map50_95 = results.metrics.map if hasattr(results, 'metrics') and results.metrics else 0.0
        
        result = {
            "model_path": os.path.join(self.output_dir, "weights", "best.pt"),
            "final_map50": final_map50,
            "final_map50_95": final_map50_95,
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
                "learning_rate": learning_rate
            },
            "config_path": yaml_config_path
        }
        
        print(f"Training completed. mAP50: {final_map50}, mAP50-95: {final_map50_95}")
        
        # Execute callback if provided
        if progress_callback:
            # Since YOLO training doesn't provide step-by-step updates easily,
            # we'll call the callback with final results
            progress_callback(epochs, epochs, {"map50": final_map50, "map50_95": final_map50_95})
        
        return result
    
    def predict(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
        
        Returns:
            Prediction result
        """
        if not self.ultralytics_available:
            raise ImportError("ultralytics is required for YOLO prediction. Install with: pip install ultralytics")
        
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(os.path.join(self.output_dir, "weights", "best.pt"))
        
        # Make prediction
        results = model(image_path, conf=conf_threshold, iou=iou_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detections.append({
                        "class_id": cls,
                        "confidence": conf,
                        "bbox": xyxy,
                        "class_name": result.names[cls] if result.names else f"class_{cls}"
                    })
        
        return {
            "detections": detections,
            "image_path": image_path,
            "num_detections": len(detections)
        }
    
    def validate_model(self, test_dataset_path: str = None):
        """
        Validate the trained model on a test dataset
        
        Args:
            test_dataset_path: Path to test dataset (uses training dataset if not provided)
        
        Returns:
            Validation metrics
        """
        if not self.ultralytics_available:
            raise ImportError("ultralytics is required for YOLO validation. Install with: pip install ultralytics")
        
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(os.path.join(self.output_dir, "weights", "best.pt"))
        
        # Use training dataset if test dataset not provided
        if test_dataset_path is None:
            test_dataset_path = self.dataset_path
        
        if test_dataset_path is None:
            raise ValueError("Test dataset path must be provided")
        
        # Validate the model
        results = model.val(data=self.create_yaml_config(test_dataset_path))
        
        return {
            "map50": results.box.map50,
            "map50_95": results.box.map,
            "precision": results.box.precision,
            "recall": results.box.recall,
            "fitness": results.fitness()
        }


if __name__ == "__main__":
    # Example usage
    print("Object Detection Trainer")
    print("=" * 25)
    
    # Example: Initialize trainer
    # trainer = ObjectDetectionTrainer(
    #     base_model="yolov8n.pt",
    #     dataset_path="./my_yolo_dataset",
    #     output_dir="./output"
    # )
    #
    # # Train the model
    # result = trainer.train(
    #     epochs=100,
    #     batch_size=16,
    #     img_size=640
    # )
    #
    # print(f"Training completed: {result}")
    
    print("Object detection trainer ready for training tasks")