"""
Model Training Routes - API endpoints for training various models
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import uuid
import os
import sys
import aiofiles
import json
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import TrainingRequest, TaskResponse, TrainingStatusResponse

router = APIRouter()


async def save_dataset(upload_file: UploadFile) -> str:
    """Save uploaded dataset and return the path"""
    dataset_dir = os.path.join(settings.UPLOAD_DIR, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_ext = os.path.splitext(upload_file.filename)[1] or ".zip"
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(dataset_dir, filename)
    
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    # If zip file, extract it
    if file_ext == ".zip":
        extract_dir = filepath.replace(".zip", "")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir
    
    return filepath


def run_image_classifier_training(task_id: str, config: dict, task_manager: TaskManager):
    """Background task for image classifier training"""
    try:
        task_manager.update_task(task_id, status="running", message="Initializing training...")
        
        from modules.training import ImageClassifierTrainer
        
        trainer = ImageClassifierTrainer(
            base_model=config["base_model"],
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"],
            num_labels=config.get("num_labels", 2)
        )
        
        def progress_callback(epoch, step, total_steps, loss, metrics):
            progress = (step / total_steps) * 100 if total_steps > 0 else 0
            task_manager.update_task(
                task_id,
                progress=progress,
                message=f"Epoch {epoch}, Step {step}/{total_steps}",
                result={
                    "current_epoch": epoch,
                    "current_step": step,
                    "total_steps": total_steps,
                    "loss": loss,
                    "metrics": metrics
                }
            )
        
        result = trainer.train(
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            progress_callback=progress_callback
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Training complete",
            result=result
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Training failed: {str(e)}"
        )


def run_object_detection_training(task_id: str, config: dict, task_manager: TaskManager):
    """Background task for object detection training (YOLO)"""
    try:
        task_manager.update_task(task_id, status="running", message="Initializing YOLO training...")
        
        from modules.training import ObjectDetectionTrainer
        
        trainer = ObjectDetectionTrainer(
            base_model=config["base_model"],
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"]
        )
        
        def progress_callback(epoch, total_epochs, metrics):
            progress = (epoch / total_epochs) * 100
            task_manager.update_task(
                task_id,
                progress=progress,
                message=f"Epoch {epoch}/{total_epochs}",
                result={
                    "current_epoch": epoch,
                    "total_epochs": total_epochs,
                    "metrics": metrics
                }
            )
        
        result = trainer.train(
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            progress_callback=progress_callback
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Training complete",
            result=result
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Training failed: {str(e)}"
        )


@router.post("/image-classifier", response_model=TaskResponse)
async def train_image_classifier(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    base_model: str = Form("google/vit-base-patch16-224"),
    epochs: int = Form(10),
    batch_size: int = Form(8),
    learning_rate: float = Form(5e-5),
    num_labels: int = Form(2),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Train an image classifier model.
    Upload a dataset as a zip file with train/val folders.
    """
    dataset_path = await save_dataset(dataset)
    
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(settings.TRAINING_OUTPUT_DIR, f"image_classifier_{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_labels": num_labels
    }
    
    task_manager.create_task(task_id, "image_classifier_training")
    
    background_tasks.add_task(run_image_classifier_training, task_id, config, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Image classifier training started"
    )


@router.post("/object-detection", response_model=TaskResponse)
async def train_object_detection(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    base_model: str = Form("yolov8n.pt"),
    epochs: int = Form(100),
    batch_size: int = Form(16),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Train an object detection model (YOLO).
    Upload a dataset in YOLO format as a zip file.
    """
    dataset_path = await save_dataset(dataset)
    
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(settings.TRAINING_OUTPUT_DIR, f"object_detection_{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size
    }
    
    task_manager.create_task(task_id, "object_detection_training")
    
    background_tasks.add_task(run_object_detection_training, task_id, config, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Object detection training started"
    )


@router.get("/status/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Get the status of a training task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    result = task.get("result", {})
    
    return TrainingStatusResponse(
        success=True,
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        current_epoch=result.get("current_epoch", 0),
        total_epochs=result.get("total_epochs", 0),
        current_step=result.get("current_step", 0),
        total_steps=result.get("total_steps", 0),
        loss=result.get("loss"),
        metrics=result.get("metrics"),
        message=task.get("message", "")
    )


@router.post("/stop/{task_id}")
async def stop_training(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Stop a running training task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    # Mark task as cancelled
    task_manager.update_task(task_id, status="cancelled", message="Training stopped by user")
    
    return {"success": True, "message": "Training stop signal sent"}


@router.get("/checkpoints/{task_id}")
async def list_checkpoints(task_id: str):
    """List available checkpoints for a training task"""
    checkpoint_dir = os.path.join(settings.TRAINING_OUTPUT_DIR, f"*_{task_id}")
    
    import glob
    dirs = glob.glob(checkpoint_dir)
    
    if not dirs:
        raise HTTPException(status_code=404, detail="No checkpoints found")
    
    checkpoints = []
    for d in dirs:
        checkpoint_files = glob.glob(os.path.join(d, "checkpoint-*"))
        checkpoints.extend(checkpoint_files)
    
    return {"checkpoints": checkpoints}
