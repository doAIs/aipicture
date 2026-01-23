"""
Video Recognition Routes - API endpoints for video classification and detection
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
from backend.schemas.request_models import VideoRecognitionRequest, RecognitionResponse, TaskResponse

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


def run_video_recognition_task(task_id: str, video_path: str, task: str, sample_frames: int, task_manager: TaskManager):
    """Background task for video recognition"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading model...")
        
        from modules.video_recognition import AdvancedVideoRecognition
        
        recognizer = AdvancedVideoRecognition()
        
        task_manager.update_task(task_id, progress=20, message="Processing video...")
        
        if task == "classify":
            results = recognizer.classify(video_path, sample_frames=sample_frames)
            task_manager.update_task(
                task_id,
                status="completed",
                progress=100,
                message="Classification complete",
                result={"classification": results}
            )
        else:
            results = recognizer.detect(video_path, sample_frames=sample_frames)
            task_manager.update_task(
                task_id,
                status="completed",
                progress=100,
                message="Detection complete",
                result={"detection": results}
            )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Recognition failed: {str(e)}"
        )


@router.post("/classify", response_model=TaskResponse)
async def classify_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sample_frames: int = Form(10),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Classify a video by sampling frames.
    Returns a task ID for tracking progress.
    """
    video_path = await save_upload_file(video)
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "video_classification")
    
    background_tasks.add_task(run_video_recognition_task, task_id, video_path, "classify", sample_frames, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Video classification started"
    )


@router.post("/detect", response_model=TaskResponse)
async def detect_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    sample_frames: int = Form(10),
    confidence_threshold: float = Form(0.5),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Detect objects in a video by sampling frames.
    Returns a task ID for tracking progress.
    """
    video_path = await save_upload_file(video)
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "video_detection")
    
    background_tasks.add_task(run_video_recognition_task, task_id, video_path, "detect", sample_frames, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Video object detection started"
    )
