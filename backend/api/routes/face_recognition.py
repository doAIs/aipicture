"""
Face Recognition Routes - API endpoints for face detection and recognition
"""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import uuid
import os
import sys
import aiofiles
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import FaceRecognitionRequest, RecognitionResponse, TaskResponse

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


@router.post("/detect")
async def detect_faces(
    image: UploadFile = File(...),
    model: str = Form("hog")
):
    """
    Detect faces in an image.
    Returns face locations.
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.face_recognition import detect_faces as do_detect
        
        results = do_detect(image_path, model=model)
        
        return {
            "success": True,
            "message": f"Detected {len(results)} faces",
            "faces": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/encode")
async def encode_face(
    image: UploadFile = File(...),
    name: str = Form(...)
):
    """
    Encode a face and save to the face database.
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.face_recognition import encode_and_save_face
        
        result = encode_and_save_face(image_path, name)
        
        return {
            "success": True,
            "message": f"Face encoded and saved as '{name}'",
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize")
async def recognize_faces(
    image: UploadFile = File(...),
    tolerance: float = Form(0.6)
):
    """
    Recognize faces in an image by comparing with known faces.
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.face_recognition import recognize_faces as do_recognize
        
        results = do_recognize(image_path, tolerance=tolerance)
        
        return {
            "success": True,
            "message": f"Found {len(results)} faces",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/known-faces")
async def list_known_faces():
    """
    List all known faces in the database.
    """
    try:
        from modules.face_recognition import list_known_faces
        
        faces = list_known_faces()
        
        return {
            "success": True,
            "count": len(faces),
            "faces": faces
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/known-faces/{name}")
async def delete_known_face(name: str):
    """
    Delete a known face from the database.
    """
    try:
        from modules.face_recognition import delete_known_face
        
        result = delete_known_face(name)
        
        return {
            "success": result,
            "message": f"Face '{name}' deleted" if result else f"Face '{name}' not found"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
