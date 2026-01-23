"""
System Routes - System status and health check endpoints
"""
from fastapi import APIRouter, Depends
from typing import List
import psutil
import torch

from backend.core.dependencies import get_model_manager, get_task_manager, ModelManager, TaskManager
from backend.schemas.request_models import SystemStatusResponse

router = APIRouter()


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    model_manager: ModelManager = Depends(get_model_manager),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get current system status including GPU, memory, and active tasks"""
    gpu_info = model_manager.get_gpu_memory_info()
    
    return SystemStatusResponse(
        status="online",
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_info.get("device_name"),
        gpu_memory_total_gb=gpu_info.get("total_gb", 0),
        gpu_memory_used_gb=gpu_info.get("allocated_gb", 0),
        gpu_memory_free_gb=gpu_info.get("free_gb", 0),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        active_tasks=len([t for t in task_manager.get_all_tasks() if t["status"] == "running"]),
        loaded_models=list(model_manager._models.keys())
    )


@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}


@router.post("/clear-models")
async def clear_models(model_manager: ModelManager = Depends(get_model_manager)):
    """Unload all cached models to free memory"""
    model_manager.unload_all()
    return {"success": True, "message": "All models unloaded"}


@router.get("/tasks")
async def get_all_tasks(task_manager: TaskManager = Depends(get_task_manager)):
    """Get all running and completed tasks"""
    return {"tasks": task_manager.get_all_tasks()}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Get a specific task by ID"""
    task = task_manager.get_task(task_id)
    if task:
        return task
    return {"error": "Task not found", "task_id": task_id}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Delete a task by ID"""
    task_manager.remove_task(task_id)
    return {"success": True, "message": f"Task {task_id} removed"}
