"""
Text to Video Routes - API endpoints for text-to-video generation
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
import uuid
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import TextToVideoRequest, GenerationResponse, TaskResponse

router = APIRouter()


def run_text_to_video_task(task_id: str, request: TextToVideoRequest, task_manager: TaskManager):
    """Background task for text-to-video generation"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading video model...")
        
        from modules.text_to_video import AdvancedTextToVideo
        
        task_manager.update_task(task_id, progress=20, message="Generating video frames...")
        
        generator = AdvancedTextToVideo()
        output_name = f"api_{task_id}"
        
        filepath = generator.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            fps=request.fps,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
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
    request: TextToVideoRequest,
    background_tasks: BackgroundTasks,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Generate a video from text prompt.
    Returns a task ID for tracking progress.
    """
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "text_to_video")
    
    background_tasks.add_task(run_text_to_video_task, task_id, request, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Video generation started (this may take several minutes)"
    )


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Get a generated video by filename"""
    filepath = os.path.join(settings.OUTPUT_DIR, "videos", "advanced_text_to_video", f"{filename}.mp4")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")
