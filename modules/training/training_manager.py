"""
Model Training Manager - Unified training orchestration
"""
import os
import sys
from typing import Dict, Callable, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class TrainingManager:
    """
    Unified training manager that handles different types of model training
    """
    
    def __init__(self):
        self.trainer_registry = {}
        self.register_default_trainers()
    
    def register_default_trainers(self):
        """Register default trainers"""
        try:
            from .image_classifier_trainer import ImageClassifierTrainer
            self.trainer_registry["image_classifier"] = ImageClassifierTrainer
        except ImportError:
            print("Warning: Image classifier trainer not available")
        
        try:
            from .object_detection_trainer import ObjectDetectionTrainer
            self.trainer_registry["object_detection"] = ObjectDetectionTrainer
        except ImportError:
            print("Warning: Object detection trainer not available")
        
        try:
            from .text_to_image_trainer import TextToImageTrainer
            self.trainer_registry["text_to_image"] = TextToImageTrainer
        except ImportError:
            print("Warning: Text-to-image trainer not available")
    
    def register_trainer(self, name: str, trainer_class):
        """Register a custom trainer"""
        self.trainer_registry[name] = trainer_class
    
    def get_trainer(self, name: str, *args, **kwargs):
        """Get an instance of a trainer"""
        if name not in self.trainer_registry:
            raise ValueError(f"Unknown trainer: {name}")
        
        trainer_class = self.trainer_registry[name]
        return trainer_class(*args, **kwargs)
    
    def list_available_trainers(self) -> list:
        """List all available trainers"""
        return list(self.trainer_registry.keys())
    
    def train_model(self, trainer_name: str, dataset_path: str, output_dir: str, 
                   base_model: str = None, **train_kwargs):
        """Train a model using the specified trainer"""
        trainer = self.get_trainer(trainer_name, base_model=base_model, 
                                  dataset_path=dataset_path, output_dir=output_dir)
        
        return trainer.train(**train_kwargs)


def create_training_config(model_type: str, **kwargs) -> Dict:
    """
    Create a training configuration for a specific model type
    
    Args:
        model_type: Type of model to train
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with training configuration
    """
    config = {
        "model_type": model_type,
        "epochs": kwargs.get("epochs", 10),
        "batch_size": kwargs.get("batch_size", 8),
        "learning_rate": kwargs.get("learning_rate", 5e-5),
        "warmup_steps": kwargs.get("warmup_steps", 100),
        "weight_decay": kwargs.get("weight_decay", 0.01),
        "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
        "fp16": kwargs.get("fp16", True),
        "eval_steps": kwargs.get("eval_steps", 500),
        "save_steps": kwargs.get("save_steps", 500),
        "logging_steps": kwargs.get("logging_steps", 100),
        "max_grad_norm": kwargs.get("max_grad_norm", 1.0),
        "seed": kwargs.get("seed", 42),
        "dataloader_num_workers": kwargs.get("dataloader_num_workers", 0),
        "save_total_limit": kwargs.get("save_total_limit", 2),
        "load_best_model_at_end": kwargs.get("load_best_model_at_end", True),
        "metric_for_best_model": kwargs.get("metric_for_best_model", "loss"),
        "greater_is_better": kwargs.get("greater_is_better", False)
    }
    
    # Add model-specific configurations
    if model_type == "image_classifier":
        config.update({
            "num_labels": kwargs.get("num_labels", 2),
            "problem_type": kwargs.get("problem_type", "single_label_classification")
        })
    elif model_type == "object_detection":
        config.update({
            "img_size": kwargs.get("img_size", 640),
            "iou_thresh": kwargs.get("iou_thresh", 0.5)
        })
    elif model_type == "text_to_image":
        config.update({
            "resolution": kwargs.get("resolution", 512),
            "center_crop": kwargs.get("center_crop", True),
            "random_flip": kwargs.get("random_flip", False)
        })
    
    return config


def validate_dataset_path(dataset_path: str, model_type: str) -> bool:
    """
    Validate that the dataset path contains the expected structure for the model type
    
    Args:
        dataset_path: Path to the dataset
        model_type: Type of model that will use the dataset
    
    Returns:
        True if dataset is valid, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return False
    
    if model_type == "image_classifier":
        # Expected structure: train/, val/ subdirectories with class folders
        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        return train_path.exists() and val_path.exists()
    
    elif model_type == "object_detection":
        # Expected structure: images/, labels/ with train/, val/, test/ splits
        images_path = dataset_path / "images"
        labels_path = dataset_path / "labels"
        return images_path.exists() and labels_path.exists()
    
    elif model_type == "text_to_image":
        # Expected structure: data.csv or images/ with captions
        csv_path = dataset_path / "data.csv"
        images_path = dataset_path / "images"
        return csv_path.exists() or images_path.exists()
    
    return True  # Default to True for other types


if __name__ == "__main__":
    # Example usage
    print("Model Training Manager")
    print("=" * 25)
    
    manager = TrainingManager()
    print(f"Available trainers: {manager.list_available_trainers()}")
    
    # Example: Create a training config
    config = create_training_config(
        "image_classifier",
        epochs=5,
        batch_size=16,
        learning_rate=2e-5,
        num_labels=10
    )
    print(f"Sample config: {config}")
    
    print("Training manager ready for model training tasks")