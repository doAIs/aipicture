"""
Unified TTS Manager
Manages multiple TTS engines with automatic fallback and lip-sync integration
"""

import logging
from pathlib import Path
from typing import Optional, Literal, Dict, Any
from enum import Enum

from .edge_tts_engine import EdgeTTSEngine
# from .cosyvoice_engine import CosyVoiceEngine  # Enable after installation
# from .gpt_sovits_engine import GPTSoVITSEngine  # Enable after installation
from .lip_sync import LipSyncEngine

logger = logging.getLogger(__name__)


class TTSEngineType(str, Enum):
    """TTS engine types"""
    EDGE_TTS = "edge-tts"
    COSY_VOICE = "cosyvoice"
    GPT_SOVITS = "gpt-sovits"


class TTSManager:
    """
    Unified TTS Manager with multi-engine support and lip-sync integration
    
    Features:
    - Multi-engine support (Edge-TTS, CosyVoice, GPT-SoVITS)
    - Automatic fallback mechanism
    - Integrated lip-sync pipeline
    - Audio-video alignment validation
    """
    
    def __init__(
        self,
        default_engine: TTSEngineType = TTSEngineType.EDGE_TTS,
        enable_lip_sync: bool = False
    ):
        """
        Initialize TTS Manager
        
        Args:
            default_engine: Default TTS engine to use
            enable_lip_sync: Enable automatic lip-sync generation
        """
        self.default_engine = default_engine
        self.enable_lip_sync = enable_lip_sync
        self.engines: Dict[TTSEngineType, Any] = {}
        self.lip_sync = None
        
        # Initialize engines
        self._init_engines()
        
        if enable_lip_sync:
            self.lip_sync = LipSyncEngine()
    
    def _init_engines(self):
        """Initialize available TTS engines"""
        # Always available: Edge-TTS
        try:
            self.engines[TTSEngineType.EDGE_TTS] = EdgeTTSEngine()
            logger.info("✅ Edge-TTS engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Edge-TTS: {e}")
        
        # Optional: CosyVoice
        try:
            # from .cosyvoice_engine import CosyVoiceEngine
            # self.engines[TTSEngineType.COSY_VOICE] = CosyVoiceEngine()
            # logger.info("✅ CosyVoice engine initialized")
            logger.info("⏭️  CosyVoice not installed (optional)")
        except Exception as e:
            logger.warning(f"⚠️  CosyVoice not available: {e}")
        
        # Optional: GPT-SoVITS
        try:
            # from .gpt_sovits_engine import GPTSoVITSEngine
            # self.engines[TTSEngineType.GPT_SOVITS] = GPTSoVITSEngine()
            # logger.info("✅ GPT-SoVITS engine initialized")
            logger.info("⏭️  GPT-SoVITS not installed (optional)")
        except Exception as e:
            logger.warning(f"⚠️  GPT-SoVITS not available: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        engine: Optional[TTSEngineType] = None,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synthesize speech with automatic engine selection and fallback
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            engine: TTS engine to use (None = use default)
            voice: Voice ID (for Edge-TTS)
            reference_audio: Reference audio for voice cloning (CosyVoice/GPT-SoVITS)
            reference_text: Reference text transcript
            **kwargs: Additional engine-specific parameters
            
        Returns:
            Dictionary with synthesis results
        """
        engine = engine or self.default_engine
        
        # Try primary engine
        try:
            result = self._synthesize_with_engine(
                engine=engine,
                text=text,
                output_path=output_path,
                voice=voice,
                reference_audio=reference_audio,
                reference_text=reference_text,
                **kwargs
            )
            return {
                "success": True,
                "engine": engine.value,
                "audio_path": result,
                "text": text
            }
        except Exception as e:
            logger.error(f"❌ {engine.value} failed: {e}")
            
            # Fallback to Edge-TTS if primary fails
            if engine != TTSEngineType.EDGE_TTS:
                logger.info("🔄 Falling back to Edge-TTS...")
                try:
                    result = self._synthesize_with_engine(
                        engine=TTSEngineType.EDGE_TTS,
                        text=text,
                        output_path=output_path,
                        voice=voice or "zh-CN-XiaoxiaoNeural",
                        **kwargs
                    )
                    return {
                        "success": True,
                        "engine": TTSEngineType.EDGE_TTS.value,
                        "audio_path": result,
                        "text": text,
                        "fallback": True
                    }
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {fallback_error}")
            
            return {
                "success": False,
                "engine": engine.value,
                "error": str(e),
                "text": text
            }
    
    def _synthesize_with_engine(
        self,
        engine: TTSEngineType,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute synthesis with specific engine"""
        if engine not in self.engines:
            raise ValueError(f"Engine {engine.value} not available")
        
        tts_engine = self.engines[engine]
        
        if engine == TTSEngineType.EDGE_TTS:
            # Edge-TTS synthesis
            return tts_engine.synthesize(
                text=text,
                output_path=output_path,
                voice=voice or "zh-CN-XiaoxiaoNeural",
                **kwargs
            )
        
        elif engine == TTSEngineType.COSY_VOICE:
            # CosyVoice synthesis with voice cloning
            return tts_engine.synthesize(
                text=text,
                output_path=output_path,
                reference_audio=reference_audio,
                reference_text=reference_text,
                **kwargs
            )
        
        elif engine == TTSEngineType.GPT_SOVITS:
            # GPT-SoVITS synthesis
            if not reference_audio:
                raise ValueError("GPT-SoVITS requires reference_audio")
            return tts_engine.synthesize(
                text=text,
                output_path=output_path,
                reference_audio=reference_audio,
                reference_text=reference_text or text,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown engine: {engine}")
    
    def synthesize_with_lip_sync(
        self,
        text: str,
        face_video: str,
        output_video: str,
        audio_output: Optional[str] = None,
        engine: Optional[TTSEngineType] = None,
        **tts_kwargs
    ) -> Dict[str, Any]:
        """
        Generate speech and create lip-synced video in one pipeline
        
        Args:
            text: Text to synthesize
            face_video: Input face video or image
            output_video: Output lip-synced video path
            audio_output: Temporary audio output path (auto-generated if None)
            engine: TTS engine to use
            **tts_kwargs: Additional TTS parameters
            
        Returns:
            Dictionary with results
        """
        if not self.enable_lip_sync:
            raise RuntimeError("Lip-sync not enabled. Set enable_lip_sync=True")
        
        # Step 1: Generate audio
        if audio_output is None:
            audio_output = str(Path(output_video).parent / "temp_audio.wav")
        
        logger.info(f"🎤 Step 1/2: Synthesizing speech...")
        audio_result = self.synthesize(
            text=text,
            output_path=audio_output,
            engine=engine,
            **tts_kwargs
        )
        
        if not audio_result["success"]:
            return audio_result
        
        # Step 2: Generate lip-synced video
        logger.info(f"👄 Step 2/2: Generating lip-synced video...")
        try:
            video_path = self.lip_sync.generate_lip_sync_video(
                face_video=face_video,
                audio_file=audio_result["audio_path"],
                output_path=output_video
            )
            
            return {
                "success": True,
                "engine": audio_result["engine"],
                "audio_path": audio_result["audio_path"],
                "video_path": video_path,
                "text": text,
                "lip_sync_enabled": True
            }
        except Exception as e:
            logger.error(f"❌ Lip-sync failed: {e}")
            return {
                "success": False,
                "error": f"Lip-sync failed: {str(e)}",
                "audio_path": audio_result["audio_path"],
                "text": text
            }
    
    def batch_synthesize_with_lip_sync(
        self,
        texts: list[str],
        face_video: str,
        output_dir: str,
        engine: Optional[TTSEngineType] = None,
        **tts_kwargs
    ) -> list[Dict[str, Any]]:
        """
        Batch process multiple texts with lip-sync
        
        Args:
            texts: List of texts to synthesize
            face_video: Input face video
            output_dir: Output directory
            engine: TTS engine to use
            **tts_kwargs: Additional TTS parameters
            
        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, text in enumerate(texts):
            logger.info(f"📝 Processing {i+1}/{len(texts)}: {text[:50]}...")
            
            video_output = str(output_dir / f"video_{i:04d}.mp4")
            audio_output = str(output_dir / f"audio_{i:04d}.wav")
            
            result = self.synthesize_with_lip_sync(
                text=text,
                face_video=face_video,
                output_video=video_output,
                audio_output=audio_output,
                engine=engine,
                **tts_kwargs
            )
            results.append(result)
        
        return results
    
    def validate_audio_video_sync(
        self,
        audio_path: str,
        video_path: str
    ) -> Dict[str, Any]:
        """
        Validate audio-video synchronization
        
        Args:
            audio_path: Path to audio file
            video_path: Path to video file
            
        Returns:
            Validation results
        """
        import librosa
        import cv2
        
        # Get audio duration
        audio, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(audio) / sr
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps
        cap.release()
        
        # Calculate sync difference
        duration_diff = abs(audio_duration - video_duration)
        is_synced = duration_diff < 0.1  # Within 100ms
        
        return {
            "audio_duration": audio_duration,
            "video_duration": video_duration,
            "duration_diff": duration_diff,
            "is_synced": is_synced,
            "sync_quality": "excellent" if duration_diff < 0.05 else 
                          "good" if duration_diff < 0.1 else
                          "acceptable" if duration_diff < 0.2 else "poor"
        }
    
    def get_available_engines(self) -> list[str]:
        """Get list of available TTS engines"""
        return [engine.value for engine in self.engines.keys()]
    
    def get_engine_info(self, engine: TTSEngineType) -> Dict[str, Any]:
        """Get information about a specific engine"""
        if engine not in self.engines:
            return {"available": False}
        
        info = {
            "available": True,
            "type": engine.value,
        }
        
        if engine == TTSEngineType.EDGE_TTS:
            info.update({
                "voice_cloning": False,
                "voices": list(EdgeTTSEngine.VOICES.keys()),
                "languages": ["zh-CN", "en-US", "ja-JP", "and 100+ more"],
                "cost": "Free"
            })
        elif engine == TTSEngineType.COSY_VOICE:
            info.update({
                "voice_cloning": True,
                "reference_audio_length": "3-10 seconds",
                "languages": ["zh", "en", "ja", "cross-lingual"],
                "cost": "Free (self-hosted)"
            })
        elif engine == TTSEngineType.GPT_SOVITS:
            info.update({
                "voice_cloning": True,
                "reference_audio_length": "1-5 seconds",
                "languages": ["zh", "en", "ja"],
                "cost": "Free (self-hosted)"
            })
        
        return info


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = TTSManager(
        default_engine=TTSEngineType.EDGE_TTS,
        enable_lip_sync=True
    )
    
    # Check available engines
    print("Available engines:", manager.get_available_engines())
    
    # Simple synthesis
    result = manager.synthesize(
        text="你好，这是一个语音合成测试。",
        output_path="output/test_audio.mp3",
        voice="zh-CN-XiaoxiaoNeural"
    )
    print("Synthesis result:", result)
    
    # Synthesis with lip-sync
    # result = manager.synthesize_with_lip_sync(
    #     text="Hello, this is a lip-sync test.",
    #     face_video="input/face_video.mp4",
    #     output_video="output/lip_synced.mp4"
    # )
    # print("Lip-sync result:", result)
