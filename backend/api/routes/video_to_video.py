"""
Video to Video Routes - API endpoints for video-to-video transformation
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
from backend.schemas.request_models import VideoToVideoRequest, GenerationResponse, TaskResponse

router = APIRouter()


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(upload_file.filename)[1] or ".mp4"
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return filepath


def run_video_to_video_task(task_id: str, video_path: str, request: VideoToVideoRequest, task_manager: TaskManager):
    """Background task for video-to-video transformation"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading model...")
        
        from modules.video_to_video import AdvancedVideoToVideo
        
        task_manager.update_task(task_id, progress=10, message="Processing video frames...")
        
        generator = AdvancedVideoToVideo()
        output_name = f"api_{task_id}"
        
        filepath = generator.generate(
            video_path=video_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            strength=request.strength,
            num_frames=request.num_frames,
            fps=request.fps,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            output_name=output_name
        )
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Video transformation complete",
            result={"output_path": filepath}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Video transformation failed: {str(e)}"
        )


@router.post("/transform", response_model=TaskResponse)
async def transform_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(None),
    strength: float = Form(0.75),
    num_frames: int = Form(30),
    fps: int = Form(8),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Transform a video based on text prompt.
    Returns a task ID for tracking progress.
    """
    video_path = await save_upload_file(video)
    
    request = VideoToVideoRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        num_frames=num_frames,
        fps=fps
    )
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "video_to_video")
    
    background_tasks.add_task(run_video_to_video_task, task_id, video_path, request, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Video transformation started (this may take many minutes)"
    )


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Get a transformed video by filename"""
    filepath = os.path.join(settings.OUTPUT_DIR, "videos", "advanced_video_to_video", f"{filename}.mp4")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")
