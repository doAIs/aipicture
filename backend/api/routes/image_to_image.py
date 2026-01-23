"""
Image to Image Routes - API endpoints for image-to-image transformation
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import uuid
import os
import sys
import aiofiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import ImageToImageRequest, GenerationResponse, TaskResponse

router = APIRouter()


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(upload_file.filename)[1] or ".png"
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return filepath


def run_image_to_image_task(task_id: str, image_path: str, prompt: str, strength: float, task_manager: TaskManager):
    """Background task for image-to-image transformation"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading model...")
        
        from modules.image_to_image import generate_image_from_image
        
        task_manager.update_task(task_id, progress=20, message="Transforming image...")
        
        output_name = f"api_{task_id}"
        image, filepath = generate_image_from_image(
            image_path=image_path,
            prompt=prompt,
            strength=strength,
            output_name=output_name
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Transformation complete",
            result={"output_path": filepath}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Transformation failed: {str(e)}"
        )


@router.post("/transform", response_model=TaskResponse)
async def transform_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(0.75),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Transform an image based on text prompt.
    Returns a task ID for tracking progress.
    """
    # Save uploaded file
    image_path = await save_upload_file(image)
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "image_to_image")
    
    background_tasks.add_task(run_image_to_image_task, task_id, image_path, prompt, strength, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Image transformation started"
    )


@router.post("/transform-sync", response_model=GenerationResponse)
async def transform_image_sync(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(0.75)
):
    """Transform an image synchronously"""
    try:
        image_path = await save_upload_file(image)
        
        from modules.image_to_image import generate_image_from_image
        
        output_name = f"api_{uuid.uuid4()}"
        result_image, filepath = generate_image_from_image(
            image_path=image_path,
            prompt=prompt,
            strength=strength,
            output_name=output_name
        )
        
        return GenerationResponse(
            success=True,
            task_id=output_name,
            output_path=filepath,
            message="Image transformed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Get a transformed image by filename"""
    filepath = os.path.join(settings.OUTPUT_DIR, "images", "basic_image_to_image", f"{filename}.png")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")
