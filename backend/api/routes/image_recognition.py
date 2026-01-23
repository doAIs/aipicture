"""
Image Recognition Routes - API endpoints for image classification and detection
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import uuid
import os
import sys
import aiofiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from backend.core.dependencies import get_task_manager, TaskManager
from backend.core.config import settings
from backend.schemas.request_models import ImageRecognitionRequest, RecognitionResponse, RecognitionResult

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


@router.post("/classify", response_model=RecognitionResponse)
async def classify_image(
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Classify an image and return top-k predictions
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.image_recognition import classify_image as do_classify
        
        results = do_classify(image_path, top_k=top_k)
        
        recognition_results = [
            RecognitionResult(label=label, confidence=conf)
            for label, conf in results
        ]
        
        return RecognitionResponse(
            success=True,
            message="Classification complete",
            results=recognition_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect", response_model=RecognitionResponse)
async def detect_objects(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    save_result: bool = Form(True)
):
    """
    Detect objects in an image
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.image_recognition import detect_objects as do_detect
        
        detections = do_detect(image_path, confidence_threshold=confidence_threshold)
        
        recognition_results = [
            RecognitionResult(
                label=det["label"],
                confidence=det["confidence"],
                bbox=det["bbox"]
            )
            for det in detections
        ]
        
        return RecognitionResponse(
            success=True,
            message=f"Detected {len(detections)} objects",
            results=recognition_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    top_k: int = Form(5),
    confidence_threshold: float = Form(0.5)
):
    """
    Perform both classification and detection on an image
    """
    try:
        image_path = await save_upload_file(image)
        
        from modules.image_recognition import classify_image as do_classify, detect_objects as do_detect
        
        # Classification
        classification_results = do_classify(image_path, top_k=top_k)
        
        # Detection
        detection_results = do_detect(image_path, confidence_threshold=confidence_threshold)
        
        return {
            "success": True,
            "classification": [
                {"label": label, "confidence": conf}
                for label, conf in classification_results
            ],
            "detection": detection_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
