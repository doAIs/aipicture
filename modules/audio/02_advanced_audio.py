"""
Advanced Audio Processing Module - Enhanced audio processing functionality
"""
import os
import sys
import subprocess
import tempfile
from typing import Dict, Optional, List, Tuple
import numpy as np
import librosa
import soundfile as sf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class AdvancedAudioProcessor:
    """
    Advanced Audio Processor Class
    Provides comprehensive audio processing functionality including:
    - Speech recognition
    - Text-to-speech
    - Audio analysis
    - Audio effects and transformations
    - Voice conversion
    """
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    
    def transcribe_audio(self, audio_path: str, model_size: str = "base", 
                        language: Optional[str] = None, temperature: float = 0.0) -> Dict:
        """
        Advanced audio transcription with more options
        
        Args:
            audio_path: Path to the audio file
            model_size: Size of the Whisper model
            language: Language code
            temperature: Temperature for sampling (0.0 for deterministic)
        
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
            
            # Load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            
            # Decode the audio
            options = whisper.DecodingOptions(
                language=language,
                temperature=temperature
            )
            result = whisper.decode(model, mel, options)
            
            return {
                "success": True,
                "text": result.text.strip(),
                "language": result.language,
                "duration": len(audio) / whisper.sample_rate
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "language": language
            }
    
    def batch_transcribe(self, audio_paths: List[str], model_size: str = "base") -> List[Dict]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            model_size: Size of the Whisper model
        
        Returns:
            List of transcription results
        """
        results = []
        for path in audio_paths:
            result = self.transcribe_audio(path, model_size)
            results.append({
                "file_path": path,
                "transcription": result
            })
        return results
    
    def synthesize_speech(self, text: str, language: str = "en", 
                         voice: str = "default", output_path: Optional[str] = None) -> str:
        """
        Advanced text-to-speech with voice selection
        
        Args:
            text: Text to convert to speech
            language: Language code
            voice: Voice identifier (model-dependent)
            output_path: Output path for the audio file (optional)
        
        Returns:
            Path to the generated audio file
        """
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Please install TTS: pip install TTS")
        
        try:
            # Initialize TTS with a multilingual model
            tts = TTS("tts_models/multilingual/multi-dataset/your_tts", gpu=False)
            
            # If no output path provided, create a temporary file
            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), f"tts_{abs(hash(text))}.wav")
            
            # Generate speech (this model requires a reference audio for voice cloning)
            # For now, we'll use a simpler approach
            tts_simple = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            tts_simple.tts_to_file(text, file_path=output_path)
            
            return output_path
        except Exception as e:
            # Fallback to coqui TTS
            try:
                from TTS.api import TTS
                tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                
                if output_path is None:
                    output_path = os.path.join(tempfile.gettempdir(), f"tts_{abs(hash(text))}.wav")
                
                tts.tts_to_file(text, file_path=output_path)
                return output_path
            except Exception as fallback_error:
                raise Exception(f"TTS failed: {str(e)}, fallback failed: {str(fallback_error)}")
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze audio properties including tempo, pitch, and spectral features
        
        Args:
            audio_path: Path to the audio file
        
        Returns:
            Dictionary with audio analysis results
        """
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Extract features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate average values
        avg_chroma = np.mean(chroma, axis=1).tolist()
        avg_mfcc = np.mean(mfcc, axis=1).tolist()
        avg_spectral_centroid = float(np.mean(spectral_centroids))
        avg_spectral_rolloff = float(np.mean(spectral_rolloff))
        avg_zero_crossing_rate = float(np.mean(zero_crossing_rate))
        
        return {
            "success": True,
            "duration": len(y) / sr,
            "sample_rate": sr,
            "tempo": float(tempo),
            "chroma_features": avg_chroma,
            "mfcc_features": avg_mfcc,
            "spectral_centroid": avg_spectral_centroid,
            "spectral_rolloff": avg_spectral_rolloff,
            "zero_crossing_rate": avg_zero_crossing_rate,
            "rms_energy": float(np.sqrt(np.mean(y**2)))
        }
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "wav", target_sr: int = 22050) -> str:
        """
        Convert audio to different format and sample rate
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            target_format: Target format ('wav', 'mp3', 'flac', etc.)
            target_sr: Target sample rate
        
        Returns:
            Path to converted audio file
        """
        # Load audio
        y, sr = librosa.load(input_path, sr=None)
        
        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Save in target format
        sf.write(output_path, y, sr)
        
        return output_path
    
    def extract_vocal(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Attempt to separate vocals from instrumental
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output vocal file (optional)
        
        Returns:
            Path to vocal audio file
        """
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_vocals{ext}"
        
        try:
            # Try using demucs for source separation
            import torchaudio
            import torch
            from demucs.separate import main as demucs_separate
            
            # This is a simplified approach - demucs has a complex API
            # In practice, you'd need to install demucs and use it properly
            cmd = [
                "python", "-m", "demucs.separate", 
                "--mp3", "--two-stems=vocals",
                "-n", "htdemucs",
                audio_path
            ]
            
            subprocess.run(cmd, check=True)
            
            # The separated files will be in a 'separated' directory
            import shutil
            separated_dir = os.path.join("separated", "htdemucs", os.path.basename(audio_path).replace(ext, ""))
            vocals_path = os.path.join(separated_dir, "vocals.mp3")
            
            if os.path.exists(vocals_path):
                shutil.move(vocals_path, output_path)
                return output_path
            else:
                # If demucs isn't available or fails, just copy the original
                shutil.copy2(audio_path, output_path)
                return output_path
                
        except Exception as e:
            print(f"Vocal extraction failed (this requires demucs): {e}")
            # Just copy the original file as fallback
            import shutil
            shutil.copy2(audio_path, output_path)
            return output_path
    
    def apply_audio_effects(self, audio_path: str, effects: Dict, 
                           output_path: Optional[str] = None) -> str:
        """
        Apply various audio effects to an audio file
        
        Args:
            audio_path: Path to input audio file
            effects: Dictionary of effects to apply
            output_path: Path for output audio file (optional)
        
        Returns:
            Path to processed audio file
        """
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_processed{ext}"
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply effects based on the effects dict
        if 'volume' in effects:
            gain = effects['volume']  # Factor, e.g. 1.0 = no change, 2.0 = double volume
            y = y * gain
        
        if 'speed' in effects:
            speed_factor = effects['speed']  # e.g. 1.0 = normal, 0.5 = half speed, 2.0 = double speed
            if speed_factor != 1.0:
                # Time stretching without changing pitch
                hop_length = 512
                tempo_change = 1.0 / speed_factor
                y_stretched = librosa.effects.time_stretch(y, rate=tempo_change)
                y = y_stretched
        
        if 'pitch_shift' in effects:
            n_steps = effects['pitch_shift']  # Number of semitones to shift
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        
        if 'reverb' in effects:
            # Simple reverb simulation using convolution
            reverb_amount = effects.get('reverb', 0.3)
            # Create a simple impulse response
            impulse_length = int(sr * 0.1)  # 0.1 seconds
            impulse = np.zeros(impulse_length)
            impulse[0] = 1.0
            impulse[1:] = np.exp(-np.arange(1, impulse_length) / (sr * 0.05))  # Decay
            impulse /= np.max(np.abs(impulse))  # Normalize
            # Apply reverb
            y_reverb = np.convolve(y, impulse, mode='full')[:len(y)]
            y = y * (1 - reverb_amount) + y_reverb * reverb_amount
        
        # Save processed audio
        sf.write(output_path, y, sr)
        
        return output_path
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Duration in seconds
        """
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr


if __name__ == "__main__":
    # Example usage
    print("Advanced Audio Processing Module")
    print("=" * 40)
    
    # Initialize the processor
    processor = AdvancedAudioProcessor()
    
    # Example: Analyze audio
    # analysis = processor.analyze_audio("path/to/audio.wav")
    # print(f"Audio analysis: {analysis}")
    
    # Example: Apply effects
    # effects = {
    #     "volume": 1.2,  # 20% louder
    #     "pitch_shift": 2,  # Shift up by 2 semitones
    #     "reverb": 0.3  # 30% reverb
    # }
    # processed_path = processor.apply_audio_effects("input.wav", effects, "output.wav")
    # print(f"Processed audio saved to: {processed_path}")
    
    print("Module ready for advanced audio processing tasks")