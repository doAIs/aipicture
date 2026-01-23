"""
Audio Processing Module - Package init file
"""
from . import (
    basic_audio as basic,
    advanced_audio as advanced
)

# Import functions/classes for easier access
try:
    from .basic_audio import (
        transcribe_audio,
        synthesize_speech,
        translate_audio,
        get_available_whisper_models,
        get_supported_languages
    )
except ImportError:
    pass

try:
    from .advanced_audio import AdvancedAudioProcessor
except ImportError:
    pass

__all__ = [
    # Basic functions
    "transcribe_audio",
    "synthesize_speech", 
    "translate_audio",
    "get_available_whisper_models",
    "get_supported_languages",
    # Advanced class
    "AdvancedAudioProcessor"
]