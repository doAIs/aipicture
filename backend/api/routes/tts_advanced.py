"""
Advanced TTS API Routes
Multi-engine TTS with voice cloning and lip-sync support
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uuid
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from modules.tts_advanced import TTSManager, TTSEngineType
from modules.tts_advanced.audio_video_aligner import AudioVideoAligner

router = APIRouter(prefix="/tts-advanced", tags=["TTS Advanced"])

# Initialize TTS Manager
tts_manager = TTSManager(
    default_engine=TTSEngineType.EDGE_TTS,
    enable_lip_sync=True
)
aligner = AudioVideoAligner()

# Output directories
OUTPUT_DIR = Path("outputs/tts_advanced")
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEO_DIR = OUTPUT_DIR / "video"
TEMP_DIR = OUTPUT_DIR / "temp"

for dir_path in [AUDIO_DIR, VIDEO_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Pydantic models
class TTSRequest(BaseModel):
    text: str
    engine: Optional[str] = "edge-tts"
    voice: Optional[str] = "zh-CN-XiaoxiaoNeural"
    rate: Optional[str] = "+0%"
    volume: Optional[str] = "+0%"
    pitch: Optional[str] = "+0Hz"


class TTSResponse(BaseModel):
    success: bool
    task_id: str
    audio_url: Optional[str] = None
    engine: Optional[str] = None
    message: Optional[str] = None


class LipSyncRequest(BaseModel):
    text: str
    engine: Optional[str] = "edge-tts"
    voice: Optional[str] = "zh-CN-XiaoxiaoNeural"


class SyncValidationResponse(BaseModel):
    is_synced: bool
    audio_duration: float
    video_duration: float
    difference_ms: float
    quality: str


@router.get("/engines")
async def list_engines():
    """Get list of available TTS engines"""
    engines = tts_manager.get_available_engines()
    
    engine_info = []
    for engine_name in engines:
        try:
            engine_type = TTSEngineType(engine_name)
            info = tts_manager.get_engine_info(engine_type)
            engine_info.append(info)
        except:
            pass
    
    return {
        "engines": engines,
        "details": engine_info
    }


@router.get("/voices")
async def list_voices(engine: str = "edge-tts"):
    """Get list of available voices for an engine"""
    if engine == "edge-tts":
        from modules.tts_advanced import EdgeTTSEngine
        voices = EdgeTTSEngine.list_voices()
        return {
            "engine": engine,
            "voices": voices[:50],  # Return first 50
            "total": len(voices)
        }
    else:
        return {
            "engine": engine,
            "message": "Voice listing not available for this engine"
        }


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text
    
    Supported engines:
    - edge-tts: Microsoft Edge TTS (100+ voices)
    - cosyvoice: Advanced voice cloning (requires installation)
    - gpt-sovits: Few-shot voice cloning (requires installation)
    """
    task_id = str(uuid.uuid4())
    output_filename = f"tts_{task_id}.mp3"
    output_path = str(AUDIO_DIR / output_filename)
    
    try:
        # Parse engine type
        engine_type = TTSEngineType(request.engine)
        
        # Synthesize
        result = tts_manager.synthesize(
            text=request.text,
            output_path=output_path,
            engine=engine_type,
            voice=request.voice,
            rate=request.rate,
            volume=request.volume,
            pitch=request.pitch
        )
        
        if result["success"]:
            return TTSResponse(
                success=True,
                task_id=task_id,
                audio_url=f"/api/tts-advanced/audio/{output_filename}",
                engine=result["engine"],
                message="Speech synthesized successfully"
            )
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Synthesis failed"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize-with-cloning")
async def synthesize_with_voice_cloning(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    engine: str = Form("cosyvoice")
):
    """
    Synthesize speech with voice cloning
    
    Requires CosyVoice or GPT-SoVITS installation
    """
    task_id = str(uuid.uuid4())
    
    # Save reference audio
    ref_audio_path = str(TEMP_DIR / f"ref_{task_id}.wav")
    with open(ref_audio_path, "wb") as f:
        content = await reference_audio.read()
        f.write(content)
    
    # Output path
    output_filename = f"cloned_{task_id}.wav"
    output_path = str(AUDIO_DIR / output_filename)
    
    try:
        engine_type = TTSEngineType(engine)
        
        result = tts_manager.synthesize(
            text=text,
            output_path=output_path,
            engine=engine_type,
            reference_audio=ref_audio_path,
            reference_text=reference_text or text
        )
        
        if result["success"]:
            return {
                "success": True,
                "task_id": task_id,
                "audio_url": f"/api/tts-advanced/audio/{output_filename}",
                "engine": result["engine"]
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesize-with-lipsync")
async def synthesize_with_lipsync(
    text: str = Form(...),
    face_video: UploadFile = File(...),
    voice: Optional[str] = Form("zh-CN-XiaoxiaoNeural"),
    engine: str = Form("edge-tts")
):
    """
    Generate speech and create lip-synced video
    
    Combines TTS + Wav2Lip for perfect audio-video sync
    """
    task_id = str(uuid.uuid4())
    
    # Save uploaded video
    video_ext = Path(face_video.filename).suffix
    input_video_path = str(TEMP_DIR / f"input_{task_id}{video_ext}")
    with open(input_video_path, "wb") as f:
        content = await face_video.read()
        f.write(content)
    
    # Output paths
    output_video = str(VIDEO_DIR / f"lipsync_{task_id}.mp4")
    audio_output = str(AUDIO_DIR / f"audio_{task_id}.wav")
    
    try:
        engine_type = TTSEngineType(engine)
        
        result = tts_manager.synthesize_with_lip_sync(
            text=text,
            face_video=input_video_path,
            output_video=output_video,
            audio_output=audio_output,
            engine=engine_type,
            voice=voice
        )
        
        if result["success"]:
            return {
                "success": True,
                "task_id": task_id,
                "video_url": f"/api/tts-advanced/video/lipsync_{task_id}.mp4",
                "audio_url": f"/api/tts-advanced/audio/audio_{task_id}.wav",
                "engine": result["engine"]
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/align-audio-video")
async def align_audio_video(
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
    method: str = Form("stretch")
):
    """
    Align audio and video duration with automatic sync correction
    
    Methods:
    - stretch: Time-stretch audio to match video (no pitch change)
    - trim: Trim longer stream
    - pad: Pad shorter stream with silence
    """
    task_id = str(uuid.uuid4())
    
    # Save uploads
    audio_path = str(TEMP_DIR / f"audio_{task_id}.wav")
    video_path = str(TEMP_DIR / f"video_{task_id}.mp4")
    
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    # Output path
    output_video = str(VIDEO_DIR / f"synced_{task_id}.mp4")
    
    try:
        # Merge with auto-alignment
        result_path = aligner.merge_audio_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_video,
            ensure_sync=True
        )
        
        # Validate result
        validation = aligner.validate_sync(audio_path, result_path)
        
        return {
            "success": True,
            "task_id": task_id,
            "video_url": f"/api/tts-advanced/video/synced_{task_id}.mp4",
            "sync_quality": validation["quality"],
            "difference_ms": validation["difference_ms"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-sync", response_model=SyncValidationResponse)
async def validate_sync(
    audio: UploadFile = File(...),
    video: UploadFile = File(...)
):
    """Validate audio-video synchronization"""
    task_id = str(uuid.uuid4())
    
    # Save uploads
    audio_path = str(TEMP_DIR / f"audio_{task_id}.wav")
    video_path = str(TEMP_DIR / f"video_{task_id}.mp4")
    
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(video_path, "wb") as f:
        f.write(await video.read())
    
    try:
        validation = aligner.validate_sync(audio_path, video_path)
        
        return SyncValidationResponse(**validation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Download generated audio file"""
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/mpeg")


@router.get("/video/{filename}")
async def get_video(filename: str):
    """Download generated video file"""
    file_path = VIDEO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(file_path, media_type="video/mp4")
