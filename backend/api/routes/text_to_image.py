"""
Text to Image Routes - API endpoints for text-to-image generation
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import TextToImageRequest, GenerationResponse, TaskResponse

router = APIRouter()


def run_text_to_image_task(task_id: str, request: TextToImageRequest, task_manager: TaskManager):
    """Background task for text-to-image generation"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading model...")
        
        from modules.text_to_image import generate_image_from_text
        
        task_manager.update_task(task_id, progress=20, message="Generating image...")
        
        # Generate the image
        output_name = f"api_{task_id}"
        image, filepath = generate_image_from_text(
            prompt=request.prompt,
            output_name=output_name,
            model_path=request.model_path or settings.LOCAL_MODEL_PATH
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Generation complete",
            result={"output_path": filepath}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Generation failed: {str(e)}"
        )


@router.post("/generate", response_model=TaskResponse)
async def generate_image(
    request: TextToImageRequest,
    background_tasks: BackgroundTasks,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Generate an image from text prompt.
    Returns a task ID for tracking the generation progress.
    """
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "text_to_image")
    
    background_tasks.add_task(run_text_to_image_task, task_id, request, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Image generation started"
    )


@router.post("/generate-sync", response_model=GenerationResponse)
async def generate_image_sync(request: TextToImageRequest):
    """
    Generate an image synchronously (blocking).
    Use for quick generation or testing.
    """
    try:
        from modules.text_to_image import generate_image_from_text
        
        output_name = f"api_{uuid.uuid4()}"
        image, filepath = generate_image_from_text(
            prompt=request.prompt,
            output_name=output_name,
            model_path=request.model_path or settings.LOCAL_MODEL_PATH
        )
        
        return GenerationResponse(
            success=True,
            task_id=output_name,
            output_path=filepath,
            message="Image generated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Get a generated image by filename"""
    filepath = os.path.join(settings.OUTPUT_DIR, "images", "basic_text_to_image", f"{filename}.png")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")
