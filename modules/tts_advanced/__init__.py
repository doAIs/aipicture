"""
"""Advanced Text-to-Speech Module
Supports multiple TTS engines: Edge-TTS, CosyVoice, GPT-SoVITS
Integrated lip-sync and audio-video alignment
"""

from .edge_tts_engine import EdgeTTSEngine
from .tts_manager import TTSManager, TTSEngineType
from .lip_sync import LipSyncEngine, RealTimeLipSync
# from .cosyvoice_engine import CosyVoiceEngine  # Enable after installation
# from .gpt_sovits_engine import GPTSoVITSEngine  # Enable after installation

__all__ = [
    'EdgeTTSEngine',
    'TTSManager',
    'TTSEngineType',
    'LipSyncEngine',
    'RealTimeLipSync',
    # 'CosyVoiceEngine',
    # 'GPTSoVITSEngine',
]
