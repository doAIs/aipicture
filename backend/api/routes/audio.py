"""
Audio Processing Routes - API endpoints for speech-to-text and text-to-speech
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
from backend.schemas.request_models import SpeechToTextRequest, TextToSpeechRequest, TaskResponse

router = APIRouter()


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file and return the path"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(upload_file.filename)[1] or ".wav"
    filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    async with aiofiles.open(filepath, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return filepath


def run_speech_to_text_task(task_id: str, audio_path: str, model_size: str, language: str, task_manager: TaskManager):
    """Background task for speech-to-text"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading Whisper model...")
        
        from modules.audio import transcribe_audio
        
        task_manager.update_task(task_id, progress=20, message="Transcribing audio...")
        
        result = transcribe_audio(audio_path, model_size=model_size, language=language)
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Transcription complete",
            result=result
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Transcription failed: {str(e)}"
        )


def run_text_to_speech_task(task_id: str, text: str, language: str, output_path: str, task_manager: TaskManager):
    """Background task for text-to-speech"""
    try:
        task_manager.update_task(task_id, status="running", message="Loading TTS model...")
        
        from modules.audio import synthesize_speech
        
        task_manager.update_task(task_id, progress=20, message="Synthesizing speech...")
        
        result_path = synthesize_speech(text, language=language, output_path=output_path)
        
        task_manager.update_task(
            task_id,
            status="completed",
            progress=100,
            message="Speech synthesis complete",
            result={"output_path": result_path}
        )
        
    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Speech synthesis failed: {str(e)}"
        )


@router.post("/speech-to-text", response_model=TaskResponse)
async def speech_to_text(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form(None),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Convert speech audio to text using Whisper.
    Returns a task ID for tracking progress.
    """
    audio_path = await save_upload_file(audio)
    
    task_id = str(uuid.uuid4())
    task_manager.create_task(task_id, "speech_to_text")
    
    background_tasks.add_task(run_speech_to_text_task, task_id, audio_path, model_size, language, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Speech-to-text processing started"
    )


@router.post("/text-to-speech", response_model=TaskResponse)
async def text_to_speech(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    language: str = Form("en"),
    task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Convert text to speech audio.
    Returns a task ID for tracking progress.
    """
    task_id = str(uuid.uuid4())
    output_path = os.path.join(settings.OUTPUT_DIR, "audio", f"tts_{task_id}.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    task_manager.create_task(task_id, "text_to_speech")
    
    background_tasks.add_task(run_text_to_speech_task, task_id, text, language, output_path, task_manager)
    
    return TaskResponse(
        success=True,
        task_id=task_id,
        message="Text-to-speech processing started"
    )


@router.get("/result/{task_id}")
async def get_audio_result(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """Get the audio result for a completed task"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] != "completed":
        return {"status": task["status"], "message": task.get("message")}
    
    if task["type"] == "text_to_speech" and task.get("result", {}).get("output_path"):
        output_path = task["result"]["output_path"]
        if os.path.exists(output_path):
            return FileResponse(output_path, media_type="audio/wav")
    
    return task.get("result", {})
