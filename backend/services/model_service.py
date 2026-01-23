"""
Model Service - Centralized model management and operations
"""
from typing import Optional, Dict, Any, List
import torch
import gc
from pathlib import Path


class ModelService:
    """Service for managing AI models lifecycle and operations"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self, model_key: str, model_loader_func, *args, **kwargs):
        """
        Load a model with the provided loader function
        Args:
            model_key: Unique identifier for the model
            model_loader_func: Function that loads the model
            *args, **kwargs: Arguments for the loader function
        """
        if model_key in self.models:
            return self.models[model_key]
        
        # Load model
        model = model_loader_func(*args, **kwargs)
        
        # Move to device
        if hasattr(model, 'to'):
            model = model.to(self.device)
        
        # Enable optimizations if available
        if hasattr(model, 'enable_attention_slicing'):
            try:
                model.enable_attention_slicing()
            except:
                pass
        
        self.models[model_key] = model
        return model
    
    def get_model(self, model_key: str):
        """Get a loaded model by key"""
        return self.models.get(model_key)
    
    def unload_model(self, model_key: str):
        """Unload a specific model to free memory"""
        if model_key in self.models:
            model = self.models.pop(model_key)
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        for model_key in list(self.models.keys()):
            self.unload_model(model_key)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model keys"""
        return list(self.models.keys())
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage information"""
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "free_gb": round(total - reserved, 2),
                "device_name": torch.cuda.get_device_name(0)
            }
        return {
            "allocated_gb": 0,
            "reserved_gb": 0,
            "total_gb": 0,
            "free_gb": 0,
            "device_name": "CPU"
        }
    
    def load_diffusion_pipeline(self, model_name: str, model_path: Optional[str] = None, 
                              pipeline_class=None, **kwargs):
        """Load a diffusion pipeline with fallback to local model"""
        from utils.modules_utils import load_model_with_fallback, load_model_from_local_file
        
        if model_path and Path(model_path).exists():
            # Load from local path
            model = load_model_from_local_file(pipeline_class, model_path, **kwargs)
        else:
            # Load from hub with fallback
            model = load_model_with_fallback(pipeline_class, model_name, **kwargs)
        
        return model
    
    def load_transformers_model(self, processor_class, model_class, model_name: str, 
                              local_model_path: Optional[str] = None, **kwargs):
        """Load a transformers model with fallback to local model"""
        from utils.modules_utils import load_transformers_model_with_fallback
        
        processor, model = load_transformers_model_with_fallback(
            processor_class, model_class, model_name, local_model_path, **kwargs
        )
        
        return processor, model
    
    def load_yolo_model(self, model_name: str = "yolov8n.pt", local_model_path: Optional[str] = None):
        """Load a YOLO model with fallback to local model"""
        from utils.modules_utils import load_yolo_model_with_fallback
        
        model = load_yolo_model_with_fallback(model_name, local_model_path)
        return model
