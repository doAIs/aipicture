"""
API Routes - Combines all route modules
"""
from fastapi import APIRouter

from .text_to_image import router as text_to_image_router
from .image_to_image import router as image_to_image_router
from .text_to_video import router as text_to_video_router
from .image_to_video import router as image_to_video_router
from .video_to_video import router as video_to_video_router
from .image_recognition import router as image_recognition_router
from .video_recognition import router as video_recognition_router
from .face_recognition import router as face_recognition_router
from .audio import router as audio_router
from .training import router as training_router
from .llm_finetuning import router as llm_finetuning_router
from .system import router as system_router

# Create main router
router = APIRouter()

# Include all sub-routers
router.include_router(system_router, prefix="/system", tags=["System"])
router.include_router(text_to_image_router, prefix="/text-to-image", tags=["Text to Image"])
router.include_router(image_to_image_router, prefix="/image-to-image", tags=["Image to Image"])
router.include_router(text_to_video_router, prefix="/text-to-video", tags=["Text to Video"])
router.include_router(image_to_video_router, prefix="/image-to-video", tags=["Image to Video"])
router.include_router(video_to_video_router, prefix="/video-to-video", tags=["Video to Video"])
router.include_router(image_recognition_router, prefix="/image-recognition", tags=["Image Recognition"])
router.include_router(video_recognition_router, prefix="/video-recognition", tags=["Video Recognition"])
router.include_router(face_recognition_router, prefix="/face-recognition", tags=["Face Recognition"])
router.include_router(audio_router, prefix="/audio", tags=["Audio Processing"])
router.include_router(training_router, prefix="/training", tags=["Model Training"])
router.include_router(llm_finetuning_router, prefix="/llm-finetuning", tags=["LLM Fine-tuning"])

__all__ = ["router"]
