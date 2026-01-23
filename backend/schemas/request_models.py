"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ==================== Enums ====================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    IMAGE_CLASSIFIER = "image_classifier"
    OBJECT_DETECTOR = "object_detector"
    FACE_RECOGNITION = "face_recognition"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    LLM = "llm"


# ==================== Base Models ====================

class BaseRequest(BaseModel):
    """Base request model with common fields"""
    pass


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = True
    message: str = "Success"
    error: Optional[str] = None


# ==================== Generation Requests ====================

class TextToImageRequest(BaseModel):
    """Request model for text-to-image generation"""
    prompt: str = Field(..., description="Text prompt describing the image to generate")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to avoid certain features")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    height: int = Field(512, ge=256, le=1024, description="Output image height")
    width: int = Field(512, ge=256, le=1024, description="Output image width")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=4, description="Number of images to generate")
    model_path: Optional[str] = Field(None, description="Custom model path")


class ImageToImageRequest(BaseModel):
    """Request model for image-to-image generation"""
    prompt: str = Field(..., description="Text prompt for transformation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    strength: float = Field(0.75, ge=0.0, le=1.0, description="Transformation strength")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Number of inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    model_path: Optional[str] = Field(None, description="Custom model path")


class TextToVideoRequest(BaseModel):
    """Request model for text-to-video generation"""
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    num_frames: int = Field(16, ge=8, le=64, description="Number of frames")
    fps: int = Field(8, ge=4, le=30, description="Frames per second")
    height: int = Field(256, ge=128, le=512, description="Video height")
    width: int = Field(256, ge=128, le=512, description="Video width")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")


class ImageToVideoRequest(BaseModel):
    """Request model for image-to-video generation"""
    num_frames: int = Field(14, ge=8, le=64, description="Number of frames")
    fps: int = Field(7, ge=4, le=30, description="Frames per second")
    motion_bucket_id: int = Field(127, ge=1, le=255, description="Motion intensity")
    noise_aug_strength: float = Field(0.02, ge=0.0, le=0.5, description="Noise augmentation")
    decode_chunk_size: int = Field(8, ge=1, le=32, description="Decode chunk size")
    seed: Optional[int] = Field(None, description="Random seed")


class VideoToVideoRequest(BaseModel):
    """Request model for video-to-video transformation"""
    prompt: str = Field(..., description="Text prompt for transformation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    strength: float = Field(0.75, ge=0.0, le=1.0, description="Transformation strength")
    num_frames: int = Field(30, ge=8, le=120, description="Number of frames to process")
    fps: int = Field(8, ge=4, le=30, description="Output FPS")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")


# ==================== Recognition Requests ====================

class ImageRecognitionRequest(BaseModel):
    """Request model for image recognition"""
    task: str = Field("classify", description="Task: classify or detect")
    top_k: int = Field(5, ge=1, le=20, description="Top K results for classification")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for detection")
    save_result: bool = Field(True, description="Save detection result with boxes")


class VideoRecognitionRequest(BaseModel):
    """Request model for video recognition"""
    task: str = Field("classify", description="Task: classify or detect")
    sample_frames: int = Field(10, ge=1, le=100, description="Number of frames to sample")
    top_k: int = Field(5, ge=1, le=20, description="Top K results")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")


class FaceRecognitionRequest(BaseModel):
    """Request model for face recognition"""
    task: str = Field("detect", description="Task: detect, encode, or recognize")
    tolerance: float = Field(0.6, ge=0.0, le=1.0, description="Recognition tolerance")
    model: str = Field("hog", description="Detection model: hog or cnn")


# ==================== Audio Requests ====================

class SpeechToTextRequest(BaseModel):
    """Request model for speech-to-text"""
    model_size: str = Field("base", description="Whisper model size: tiny, base, small, medium, large")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'zh')")
    task: str = Field("transcribe", description="Task: transcribe or translate")


class TextToSpeechRequest(BaseModel):
    """Request model for text-to-speech"""
    text: str = Field(..., description="Text to synthesize")
    language: str = Field("en", description="Language code")
    speaker: Optional[str] = Field(None, description="Speaker ID for multi-speaker models")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")


# ==================== Training Requests ====================

class TrainingRequest(BaseModel):
    """Request model for model training"""
    model_type: ModelType = Field(..., description="Type of model to train")
    base_model: str = Field(..., description="Base model path or name")
    dataset_path: str = Field(..., description="Path to training dataset")
    output_dir: str = Field(..., description="Output directory for checkpoints")
    
    # Training hyperparameters
    epochs: int = Field(10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(8, ge=1, le=64, description="Batch size")
    learning_rate: float = Field(5e-5, ge=1e-7, le=1e-2, description="Learning rate")
    warmup_steps: int = Field(100, ge=0, description="Warmup steps")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, ge=1, le=64, description="Gradient accumulation")
    
    # Evaluation
    eval_steps: int = Field(500, ge=1, description="Evaluation interval")
    save_steps: int = Field(500, ge=1, description="Checkpoint save interval")
    
    # Mixed precision
    fp16: bool = Field(True, description="Use FP16 training")
    
    # Additional options
    resume_from_checkpoint: Optional[str] = Field(None, description="Resume from checkpoint")


class LoRATrainingRequest(BaseModel):
    """Request model for LoRA fine-tuning"""
    base_model: str = Field(..., description="Base model path or name")
    dataset_path: str = Field(..., description="Path to training dataset")
    output_dir: str = Field(..., description="Output directory")
    
    # LoRA configuration
    lora_r: int = Field(8, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, le=512, description="LoRA alpha")
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA")
    
    # Training hyperparameters
    epochs: int = Field(3, ge=1, le=50, description="Training epochs")
    batch_size: int = Field(4, ge=1, le=32, description="Batch size")
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2, description="Learning rate")
    gradient_accumulation_steps: int = Field(4, ge=1, le=64, description="Gradient accumulation")
    
    # Quantization
    use_4bit: bool = Field(False, description="Use 4-bit quantization (QLoRA)")
    use_8bit: bool = Field(False, description="Use 8-bit quantization")
    
    # Evaluation
    eval_steps: int = Field(100, ge=1, description="Evaluation interval")
    save_steps: int = Field(100, ge=1, description="Save interval")


# ==================== Response Models ====================

class GenerationResponse(BaseResponse):
    """Response model for generation tasks"""
    task_id: str = Field(..., description="Task ID for tracking")
    output_path: Optional[str] = Field(None, description="Path to generated file")
    output_url: Optional[str] = Field(None, description="URL to access generated file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RecognitionResult(BaseModel):
    """Single recognition result"""
    label: str
    confidence: float
    bbox: Optional[List[float]] = None  # For detection: [x1, y1, x2, y2]


class RecognitionResponse(BaseResponse):
    """Response model for recognition tasks"""
    results: List[RecognitionResult] = Field(default_factory=list)
    output_path: Optional[str] = Field(None, description="Path to annotated output")


class TrainingStatusResponse(BaseResponse):
    """Response model for training status"""
    task_id: str
    status: TaskStatus
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    eta_seconds: Optional[int] = None


class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    status: str = "online"
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_total_gb: float = 0
    gpu_memory_used_gb: float = 0
    gpu_memory_free_gb: float = 0
    cpu_percent: float = 0
    memory_percent: float = 0
    active_tasks: int = 0
    loaded_models: List[str] = Field(default_factory=list)


class TaskResponse(BaseResponse):
    """Response model for async task creation"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Task created successfully"


class FileUploadResponse(BaseResponse):
    """Response model for file uploads"""
    filename: str
    file_path: str
    file_size: int
    content_type: str
