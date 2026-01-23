"""
LLM Fine-tuning Routes - API endpoints for LoRA/QLoRA fine-tuning
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import uuid
import os
import sys
import aiofiles
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import LoRATrainingRequest, TaskResponse, TrainingStatusResponse

router = APIRouter()


async def save_dataset(upload_file: UploadFile) -> str:
    """Save uploaded dataset and return the path"""
    dataset_dir = os.path.join(settings.UPLOAD_DIR, "datasets", "llm")
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_ext = os.path.splitext(upload_file.filename)[1] or ".json"
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(dataset_dir, filename)
    
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return filepath


def run_lora_training(task_id: str, config: dict, task_manager: TaskManager):
    """Background task for LoRA fine-tuning"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading base model...")
        
        from modules.llm_finetuning import LoRATrainer
        
        trainer = LoRATrainer(
            base_model=config["base_model"],
            dataset_path=config["dataset_path"],
            output_dir=config["output_dir"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config.get("target_modules"),
            use_4bit=config.get("use_4bit", False),
            use_8bit=config.get("use_8bit", False)
        )
        
        task_manager.update_task(task_id, progress=10, message="Model loaded, starting training...")
        
        def progress_callback(step, total_steps, loss, metrics):
            progress = (step / total_steps) * 100 if total_steps > 0 else 0
            task_manager.update_task(
                task_id,
                progress=progress,
                message=f"Step {step}/{total_steps}, Loss: {loss:.4f}",
                result={
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
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            progress_callback=progress_callback
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="LoRA training complete",
            result=result
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Training failed: {str(e)}"
        )


@router.post("/lora", response_model=TaskResponse)
async def train_lora(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    base_model: str = Form(...),
    lora_r: int = Form(8),
    lora_alpha: int = Form(16),
    lora_dropout: float = Form(0.05),
    epochs: int = Form(3),
    batch_size: int = Form(4),
    learning_rate: float = Form(2e-4),
    gradient_accumulation_steps: int = Form(4),
    use_4bit: bool = Form(False),
    use_8bit: bool = Form(False),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Fine-tune a language model using LoRA.
    Upload a dataset in JSON format (instruction-response pairs).
    """
    dataset_path = await save_dataset(dataset)
    
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(settings.TRAINING_OUTPUT_DIR, f"lora_{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_4bit": use_4bit,
        "use_8bit": use_8bit
    }
    
    task_manager.create_task(task_id, "lora_training")
    
    background_tasks.add_task(run_lora_training, task_id, config, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="LoRA training started"
    )


@router.post("/qlora", response_model=TaskResponse)
async def train_qlora(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    base_model: str = Form(...),
    lora_r: int = Form(64),
    lora_alpha: int = Form(16),
    lora_dropout: float = Form(0.1),
    epochs: int = Form(3),
    batch_size: int = Form(4),
    learning_rate: float = Form(2e-4),
    gradient_accumulation_steps: int = Form(4),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Fine-tune a language model using QLoRA (4-bit quantization).
    More memory efficient than standard LoRA.
    """
    dataset_path = await save_dataset(dataset)
    
    task_id = str(uuid.uuid4())
    output_dir = os.path.join(settings.TRAINING_OUTPUT_DIR, f"qlora_{task_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_4bit": True,
        "use_8bit": False
    }
    
    task_manager.create_task(task_id, "qlora_training")
    
    background_tasks.add_task(run_lora_training, task_id, config, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="QLoRA training started"
    )


@router.get("/status/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Get the status of an LLM training task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    result = task.get("result", {})
    
    return TrainingStatusResponse(
        success=True,
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        current_step=result.get("current_step", 0),
        total_steps=result.get("total_steps", 0),
        loss=result.get("loss"),
        metrics=result.get("metrics"),
        message=task.get("message", "")
    )


@router.post("/merge/{task_id}")
async def merge_adapter(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Merge LoRA adapter with base model"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Training task not found")
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not completed yet")
    
    try:
        from modules.llm_finetuning import merge_lora_adapter
        
        output_dir = task.get("result", {}).get("output_dir")
        if not output_dir:
            raise HTTPException(status_code=400, detail="Output directory not found")
        
        merged_path = merge_lora_adapter(output_dir)
        
        return {
            "success": True,
            "message": "Adapter merged successfully",
            "merged_model_path": merged_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters")
async def list_adapters():
    """List all trained LoRA adapters"""
    import glob
    
    adapter_dirs = glob.glob(os.path.join(settings.TRAINING_OUTPUT_DIR, "lora_*"))
    adapter_dirs.extend(glob.glob(os.path.join(settings.TRAINING_OUTPUT_DIR, "qlora_*")))
    
    adapters = []
    for d in adapter_dirs:
        adapter_config = os.path.join(d, "adapter_config.json")
        if os.path.exists(adapter_config):
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            adapters.append({
                "path": d,
                "name": os.path.basename(d),
                "config": config
            })
    
    return {"adapters": adapters}
