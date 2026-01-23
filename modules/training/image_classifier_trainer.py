"""
Image Classifier Trainer - Fine-tune image classification models
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, Optional, Callable, List
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize
)
import evaluate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class ImageClassifierTrainer:
    """
    Trainer for image classification models using Hugging Face Transformers
    Supports models like ViT, ResNet, EfficientNet, etc.
    """
    
    def __init__(self, base_model: str = "google/vit-base-patch16-224", 
                 dataset_path: str = None, output_dir: str = "./image_classifier_output",
                 num_labels: int = 2):
        """
        Initialize the image classifier trainer
        
        Args:
            base_model: Pre-trained model name or path
            dataset_path: Path to the dataset directory
            output_dir: Output directory for trained model
            num_labels: Number of classes in the classification task
        """
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.device = get_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(base_model)
        self.model = AutoModelForImageClassification.from_pretrained(
            base_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # Handle mismatched classifier layers
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize transforms
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Get training transforms"""
        return Compose([
            RandomResizedCrop(size=(self.feature_extractor.size['height'], self.feature_extractor.size['width'])),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        ])
    
    def _get_val_transforms(self):
        """Get validation transforms"""
        return Compose([
            RandomResizedCrop(size=(self.feature_extractor.size['height'], self.feature_extractor.size['width'])),
            ToTensor(),
            Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        ])
    
    def preprocess_train(self, example_batch):
        """Preprocess training examples"""
        images = [self.train_transforms(img.convert("RGB")) for img in example_batch['image']]
        return {'pixel_values': images}
    
    def preprocess_val(self, example_batch):
        """Preprocess validation examples"""
        images = [self.val_transforms(img.convert("RGB")) for img in example_batch['image']]
        return {'pixel_values': images}
    
    def load_dataset(self, dataset_path: str = None):
        """
        Load dataset from directory structure
        Expected structure:
        dataset_path/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── val/
            ├── class1/
            ├── class2/
            └── ...
        """
        if dataset_path:
            self.dataset_path = dataset_path
        
        if not self.dataset_path:
            raise ValueError("Dataset path must be provided")
        
        from datasets import load_dataset
        
        # Load train and validation datasets
        train_dataset = load_dataset("imagefolder", data_dir=os.path.join(self.dataset_path, "train"))
        val_dataset = load_dataset("imagefolder", data_dir=os.path.join(self.dataset_path, "val"))
        
        # Preprocess datasets
        train_dataset = train_dataset.with_transform(self.preprocess_train)
        val_dataset = val_dataset.with_transform(self.preprocess_val)
        
        return train_dataset['train'], val_dataset['train']
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    def train(self, 
              epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              gradient_accumulation_steps: int = 1,
              fp16: bool = True,
              eval_steps: int = 500,
              save_steps: int = 500,
              logging_steps: int = 100,
              max_grad_norm: float = 1.0,
              seed: int = 42,
              dataloader_num_workers: int = 0,
              save_total_limit: int = 2,
              load_best_model_at_end: bool = True,
              metric_for_best_model: str = "accuracy",
              greater_is_better: bool = True,
              progress_callback: Optional[Callable] = None):
        """
        Train the image classification model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            gradient_accumulation_steps: Gradient accumulation steps
            fp16: Use mixed precision training
            eval_steps: Evaluation steps
            save_steps: Save steps
            logging_steps: Logging steps
            max_grad_norm: Max gradient norm
            seed: Random seed
            dataloader_num_workers: Number of workers for data loading
            save_total_limit: Limit of total saved checkpoints
            load_best_model_at_end: Load best model at end
            metric_for_best_model: Metric for selecting best model
            greater_is_better: Whether higher metric values are better
            progress_callback: Callback for progress updates
        
        Returns:
            Training result
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load datasets
        train_dataset, val_dataset = self.load_dataset()
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            logging_steps=logging_steps,
            max_grad_norm=max_grad_norm,
            dataloader_num_workers=dataloader_num_workers,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to=None,  # Disable reporting to external services
            push_to_hub=False,
            disable_tqdm=True,  # We'll handle progress manually
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if load_best_model_at_end else []
        )
        
        # Train the model
        print(f"Starting training for {epochs} epochs...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        trainer.save_state()
        
        # Compute final metrics
        metrics = trainer.evaluate()
        print(f"Final metrics: {metrics}")
        
        # Prepare result
        result = {
            "model_path": self.output_dir,
            "train_loss": train_result.training_loss,
            "eval_metrics": metrics,
            "training_args": training_args.to_dict()
        }
        
        # Execute callback if provided
        if progress_callback:
            progress_callback(epochs, 0, 0, train_result.training_loss, metrics)
        
        return result
    
    def predict(self, image_path: str):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image
        
        Returns:
            Prediction result
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = self.val_transforms(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Get label names if available
        id2label = self.model.config.id2label
        predicted_label = id2label[predicted_class_idx] if id2label else f"class_{predicted_class_idx}"
        
        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "all_probabilities": probabilities[0].tolist(),
            "predicted_class_idx": predicted_class_idx
        }


if __name__ == "__main__":
    # Example usage
    print("Image Classifier Trainer")
    print("=" * 25)
    
    # Example: Initialize trainer
    # trainer = ImageClassifierTrainer(
    #     base_model="google/vit-base-patch16-224",
    #     dataset_path="./my_dataset",
    #     output_dir="./output",
    #     num_labels=2
    # )
    #
    # # Train the model
    # result = trainer.train(
    #     epochs=3,
    #     batch_size=16,
    #     learning_rate=2e-5
    # )
    #
    # print(f"Training completed: {result}")
    
    print("Image classifier trainer ready for training tasks")