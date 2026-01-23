"""
Basic Audio Processing Module - Speech recognition and text-to-speech
"""
import os
import sys
from typing import Dict, Optional
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


def transcribe_audio(audio_path: str, model_size: str = "base", language: Optional[str] = None) -> Dict:
    """
    Transcribe audio to text using OpenAI Whisper
    
    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en', 'zh', 'fr', etc.)
    
    Returns:
        Dictionary with transcription result
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install whisper: pip install openai-whisper")
    
    try:
        # Load the model
        model = whisper.load_model(model_size)
        
        # Transcribe the audio
        result = model.transcribe(audio_path, language=language)
        
        return {
            "success": True,
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "duration": result.get("duration", 0)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "segments": [],
            "language": language
        }


def synthesize_speech(text: str, language: str = "en", output_path: Optional[str] = None) -> str:
    """
    Synthesize speech from text using a TTS model
    
    Args:
        text: Text to convert to speech
        language: Language code
        output_path: Output path for the audio file (optional)
    
    Returns:
        Path to the generated audio file
    """
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("Please install TTS: pip install TTS")
    
    try:
        # Initialize TTS with a default model
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # If no output path provided, create a temporary file
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"tts_{hash(text)}.wav")
        
        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=None,  # Use default voice
            language=language
        )
        
        return output_path
    except Exception as e:
        # Fallback to a simpler model if xtts_v2 fails
        try:
            from TTS.api import TTS
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            
            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), f"tts_{hash(text)}.wav")
            
            tts.tts_to_file(text, file_path=output_path)
            return output_path
        except Exception as fallback_error:
            raise Exception(f"TTS failed: {str(e)}, fallback failed: {str(fallback_error)}")


def translate_audio(audio_path: str, model_size: str = "base") -> Dict:
    """
    Translate audio to English using OpenAI Whisper
    
    Args:
        audio_path: Path to the audio file
        model_size: Size of the Whisper model
    
    Returns:
        Dictionary with translation result
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Please install whisper: pip install openai-whisper")
    
    try:
        # Load the model
        model = whisper.load_model(model_size)
        
        # Translate the audio to English
        result = model.transcribe(audio_path, task="translate")
        
        return {
            "success": True,
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": "en",  # Always English for translation
            "duration": result.get("duration", 0)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "segments": [],
            "language": "en"
        }


def get_available_whisper_models() -> list:
    """
    Get list of available Whisper model sizes
    
    Returns:
        List of model size strings
    """
    return ["tiny", "base", "small", "medium", "large", "large-v2"]


def get_supported_languages() -> dict:
    """
    Get dictionary of supported languages for Whisper
    
    Returns:
        Dictionary mapping language codes to language names
    """
    return {
        "en": "English",
        "zh": "Chinese",
        "de": "German", 
        "es": "Spanish",
        "ru": "Russian",
        "ko": "Korean",
        "fr": "French",
        "ja": "Japanese",
        "pt": "Portuguese",
        "tr": "Turkish",
        "pl": "Polish",
        "ca": "Catalan",
        "nl": "Dutch",
        "ar": "Arabic",
        "sv": "Swedish",
        "it": "Italian",
        "id": "Indonesian",
        "hi": "Hindi",
        "fi": "Finnish",
        "vi": "Vietnamese",
        "he": "Hebrew",
        "uk": "Ukrainian",
        "el": "Greek",
        "ms": "Malay",
        "cs": "Czech",
        "ro": "Romanian",
        "da": "Danish",
        "hu": "Hungarian",
        "ta": "Tamil",
        "no": "Norwegian",
        "th": "Thai",
        "ur": "Urdu",
        "hr": "Croatian",
        "bg": "Bulgarian",
        "lt": "Lithuanian",
        "la": "Latin",
        "mi": "Maori",
        "ml": "Malayalam",
        "cy": "Welsh",
        "sk": "Slovak",
        "te": "Telugu",
        "fa": "Persian",
        "lv": "Latvian",
        "bn": "Bengali",
        "sr": "Serbian",
        "az": "Azerbaijani",
        "sl": "Slovenian",
        "kn": "Kannada",
        "et": "Estonian",
        "mk": "Macedonian",
        "br": "Breton",
        "eu": "Basque",
        "is": "Icelandic",
        "hy": "Armenian",
        "ne": "Nepali",
        "mn": "Mongolian",
        "bs": "Bosnian",
        "kk": "Kazakh",
        "sq": "Albanian",
        "sw": "Swahili",
        "gl": "Galician",
        "mr": "Marathi",
        "pa": "Punjabi",
        "si": "Sinhala",
        "km": "Khmer",
        "sn": "Shona",
        "yo": "Yoruba",
        "so": "Somali",
        "af": "Afrikaans",
        "oc": "Occitan",
        "ka": "Georgian",
        "be": "Belarusian",
        "tg": "Tajik",
        "sd": "Sindhi",
        "gu": "Gujarati",
        "am": "Amharic",
        "yi": "Yiddish",
        "lo": "Lao",
        "uz": "Uzbek",
        "fo": "Faroese",
        "ht": "Haitian Creole",
        "ps": "Pashto",
        "tk": "Turkmen",
        "nn": "Nynorsk",
        "mt": "Maltese",
        "sa": "Sanskrit",
        "lb": "Luxembourgish",
        "my": "Myanmar",
        "bo": "Tibetan",
        "tl": "Tagalog",
        "mg": "Malagasy",
        "as": "Assamese",
        "tt": "Tatar",
        "haw": "Hawaiian",
        "ln": "Lingala",
        "ha": "Hausa",
        "ba": "Bashkir",
        "jw": "Javanese",
        "su": "Sundanese"
    }


if __name__ == "__main__":
    # Example usage
    print("Basic Audio Processing Module")
    print("=" * 35)
    
    # Example: Transcribe audio
    # result = transcribe_audio("path/to/audio.wav", model_size="base")
    # print(f"Transcription: {result}")
    
    # Example: Synthesize speech
    # audio_path = synthesize_speech("Hello, this is a test of the text to speech system.", language="en")
    # print(f"Generated audio: {audio_path}")
    
    print("Module ready for audio processing tasks")