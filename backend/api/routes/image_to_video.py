"""
Image to Video Routes - API endpoints for image-to-video generation
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
from backend.schemas.request_models import ImageToVideoRequest, GenerationResponse, TaskResponse

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


def run_image_to_video_task(task_id: str, image_path: str, request: ImageToVideoRequest, task_manager: TaskManager):
    """Background task for image-to-video generation"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading SVD model...")
        
        from modules.image_to_video import AdvancedImageToVideo
        
        task_manager.update_task(task_id, progress=20, message="Generating video from image...")
        
        generator = AdvancedImageToVideo()
        output_name = f"api_{task_id}"
        
        filepath = generator.generate(
            image_path=image_path,
            num_frames=request.num_frames,
            fps=request.fps,
            motion_bucket_id=request.motion_bucket_id,
            output_name=output_name
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Video generation complete",
            result={"output_path": filepath}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Video generation failed: {str(e)}"
        )


@router.post("/generate", response_model=TaskResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    num_frames: int = Form(14),
    fps: int = Form(7),
    motion_bucket_id: int = Form(127),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Generate a video from an input image.
    Returns a task ID for tracking progress.
    """
    image_path = await save_upload_file(image)
    
    request = ImageToVideoRequest(
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id
    )
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "image_to_video")
    
    background_tasks.add_task(run_image_to_video_task, task_id, image_path, request, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Video generation started (this may take several minutes)"
    )


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Get a generated video by filename"""
    filepath = os.path.join(settings.OUTPUT_DIR, "videos", "advanced_image_to_video", f"{filename}.mp4")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")
