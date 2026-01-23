"""
Shared dependencies for FastAPI routes
"""
import torch
from typing import Optional
from functools import lru_cache


@lru_cache()
def get_device() -> str:
    """Get the available compute device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU (GPU acceleration not available)")
    return device


class ModelManager:
    """
    Centralized model management to avoid loading models multiple times.
    Uses lazy loading to only load models when needed.
    """
    
    def __init__(self):
        self._models = {}
        self._device = get_device()
    
    @property
    def device(self) -> str:
        return self._device
    
    def get_model(self, model_key: str):
        """Get a cached model by key"""
        return self._models.get(model_key)
    
    def set_model(self, model_key: str, model):
        """Cache a model by key"""
        self._models[model_key] = model
    
    def has_model(self, model_key: str) -> bool:
        """Check if a model is cached"""
        return model_key in self._models
    
    def unload_model(self, model_key: str):
        """Unload a model to free memory"""
        if model_key in self._models:
            del self._models[model_key]
            if self._device == "cuda":
                torch.cuda.empty_cache()
    
    def unload_all(self):
        """Unload all models to free memory"""
        self._models.clear()
        if self._device == "cuda":
            torch.cuda.empty_cache()
    
    def get_gpu_memory_info(self) -> dict:
        """Get GPU memory usage information"""
        if self._device == "cuda" and torch.cuda.is_available():
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


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Task tracking for background jobs
class TaskManager:
    """Manage background tasks and their status"""
    
    def __init__(self):
        self._tasks = {}
    
    def create_task(self, task_id: str, task_type: str) -> dict:
        """Create a new task entry"""
        task = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "progress": 0,
            "message": "Task created",
            "result": None,
            "error": None
        }
        self._tasks[task_id] = task
        return task
    
    def update_task(self, task_id: str, **kwargs):
        """Update task properties"""
        if task_id in self._tasks:
            self._tasks[task_id].update(kwargs)
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def remove_task(self, task_id: str):
        """Remove a task"""
        if task_id in self._tasks:
            del self._tasks[task_id]
    
    def get_all_tasks(self) -> list:
        """Get all tasks"""
        return list(self._tasks.values())


# Singleton task manager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get the global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
