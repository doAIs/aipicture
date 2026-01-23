"""
API Configuration - Centralized settings for the FastAPI backend
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    API_TITLE: str = "AI Multimedia Platform API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Professional AI multimedia processing platform with training support"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    
    # Device Settings
    CUDA_AVAILABLE: bool = True
    DEVICE: str = "cuda"  # cuda or cpu
    
    # Model Paths (inherit from existing config)
    LOCAL_MODEL_PATH: Optional[str] = os.getenv("LOCAL_MODEL_PATH", "F:\\modules\\sd")
    LOCAL_VIDEO_MODEL_PATH: Optional[str] = os.getenv("LOCAL_VIDEO_MODEL_PATH", "F:\\modules\\text-to-video\\damo")
    LOCAL_IMAGE_RECOGNITION_MODEL_PATH: Optional[str] = os.getenv("LOCAL_IMAGE_RECOGNITION_MODEL_PATH")
    LOCAL_OBJECT_DETECTION_MODEL_PATH: Optional[str] = os.getenv("LOCAL_OBJECT_DETECTION_MODEL_PATH")
    
    # Face Recognition Settings
    FACE_RECOGNITION_MODEL: str = "hog"  # hog or cnn
    FACE_DB_PATH: str = "data/faces"
    
    # Audio Settings
    WHISPER_MODEL: str = "base"  # tiny, base, small, medium, large
    TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    
    # Training Settings
    TRAINING_OUTPUT_DIR: str = "training_outputs"
    CHECKPOINTS_DIR: str = "checkpoints"
    MAX_TRAINING_EPOCHS: int = 100
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_LEARNING_RATE: float = 5e-5
    
    # LLM Fine-tuning Settings
    LLM_MODEL_PATH: Optional[str] = None
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 100
    
    # Generation Defaults
    DEFAULT_STEPS: int = 50
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_HEIGHT: int = 512
    DEFAULT_WIDTH: int = 512
    DEFAULT_VIDEO_FRAMES: int = 16
    DEFAULT_VIDEO_FPS: int = 8
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()

# Create necessary directories
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TRAINING_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(settings.FACE_DB_PATH, exist_ok=True)
