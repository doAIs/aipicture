"""
Schemas module - Pydantic models for API requests and responses
"""
from .request_models import *

__all__ = [
    # Generation requests
    "TextToImageRequest",
    "ImageToImageRequest",
    "TextToVideoRequest",
    "ImageToVideoRequest",
    "VideoToVideoRequest",
    # Recognition requests
    "ImageRecognitionRequest",
    "VideoRecognitionRequest",
    "FaceRecognitionRequest",
    # Audio requests
    "SpeechToTextRequest",
    "TextToSpeechRequest",
    # Training requests
    "TrainingRequest",
    "LoRATrainingRequest",
    # Responses
    "GenerationResponse",
    "RecognitionResponse",
    "TrainingStatusResponse",
    "SystemStatusResponse",
]
