"""
Audio-Video Alignment Module
Ensures perfect synchronization between audio and video for lip-sync
"""

import numpy as np
import librosa
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import subprocess


class AudioVideoAligner:
    """
    Advanced audio-video alignment and synchronization
    
    Features:
    - Precise duration matching
    - Audio stretching/compression without pitch change
    - Video speed adjustment
    - Frame-level synchronization
    - Quality validation
    """
    
    def __init__(self):
        """Initialize aligner"""
        self.sample_rate = 16000
        self.target_fps = 25
    
    def align_audio_to_video(
        self,
        audio_path: str,
        video_path: str,
        output_audio: str,
        method: str = "stretch"
    ) -> Tuple[str, dict]:
        """
        Align audio duration to match video duration
        
        Args:
            audio_path: Input audio file
            video_path: Reference video file
            output_audio: Output aligned audio file
            method: Alignment method ("stretch", "trim", "pad")
            
        Returns:
            Tuple of (output_path, alignment_info)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio_duration = len(audio) / sr
        
        # Get video duration
        video_duration = self._get_video_duration(video_path)
        
        # Calculate alignment
        duration_diff = video_duration - audio_duration
        
        if abs(duration_diff) < 0.05:  # Within 50ms, no adjustment needed
            import shutil
            shutil.copy(audio_path, output_audio)
            return output_audio, {
                "method": "none",
                "original_duration": audio_duration,
                "target_duration": video_duration,
                "difference": duration_diff,
                "aligned": True
            }
        
        # Apply alignment method
        if method == "stretch":
            # Time-stretch audio without changing pitch
            stretch_factor = video_duration / audio_duration
            aligned_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
        
        elif method == "trim":
            # Trim audio if too long
            if duration_diff < 0:
                target_samples = int(video_duration * sr)
                aligned_audio = audio[:target_samples]
            else:
                aligned_audio = audio
        
        elif method == "pad":
            # Pad with silence if too short
            if duration_diff > 0:
                pad_samples = int(duration_diff * sr)
                aligned_audio = np.pad(audio, (0, pad_samples), mode='constant')
            else:
                aligned_audio = audio
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save aligned audio
        Path(output_audio).parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(output_audio, aligned_audio, sr)
        
        return output_audio, {
            "method": method,
            "original_duration": audio_duration,
            "target_duration": video_duration,
            "aligned_duration": len(aligned_audio) / sr,
            "difference": duration_diff,
            "stretch_factor": stretch_factor if method == "stretch" else 1.0,
            "aligned": True
        }
    
    def align_video_to_audio(
        self,
        video_path: str,
        audio_path: str,
        output_video: str,
        method: str = "speed"
    ) -> Tuple[str, dict]:
        """
        Align video duration to match audio duration
        
        Args:
            video_path: Input video file
            audio_path: Reference audio file
            output_video: Output aligned video file
            method: Alignment method ("speed", "trim", "loop")
            
        Returns:
            Tuple of (output_path, alignment_info)
        """
        # Get durations
        video_duration = self._get_video_duration(video_path)
        audio, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(audio) / sr
        
        duration_diff = audio_duration - video_duration
        
        if abs(duration_diff) < 0.05:
            import shutil
            shutil.copy(video_path, output_video)
            return output_video, {
                "method": "none",
                "aligned": True
            }
        
        # Use FFmpeg for video adjustment
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        
        if method == "speed":
            # Adjust video speed to match audio
            speed_factor = video_duration / audio_duration
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-filter:v", f"setpts={speed_factor}*PTS",
                "-an",  # Remove audio
                output_video
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_video, {
                "method": method,
                "speed_factor": speed_factor,
                "original_duration": video_duration,
                "target_duration": audio_duration,
                "aligned": True
            }
        
        elif method == "trim":
            # Trim video to audio length
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", str(audio_duration),
                "-c", "copy",
                output_video
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_video, {
                "method": method,
                "trimmed_duration": audio_duration,
                "aligned": True
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def validate_sync(
        self,
        audio_path: str,
        video_path: str,
        tolerance_ms: float = 100.0
    ) -> dict:
        """
        Validate audio-video synchronization
        
        Args:
            audio_path: Audio file path
            video_path: Video file path
            tolerance_ms: Acceptable difference in milliseconds
            
        Returns:
            Validation results
        """
        # Get durations
        audio, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(audio) / sr
        
        video_duration = self._get_video_duration(video_path)
        
        # Calculate difference
        diff_ms = abs(audio_duration - video_duration) * 1000
        is_synced = diff_ms <= tolerance_ms
        
        # Quality rating
        if diff_ms < 50:
            quality = "excellent"
        elif diff_ms < 100:
            quality = "good"
        elif diff_ms < 200:
            quality = "acceptable"
        else:
            quality = "poor"
        
        return {
            "is_synced": is_synced,
            "audio_duration": audio_duration,
            "video_duration": video_duration,
            "difference_ms": diff_ms,
            "tolerance_ms": tolerance_ms,
            "quality": quality,
            "needs_alignment": not is_synced
        }
    
    def extract_phoneme_timestamps(
        self,
        audio_path: str,
        transcript: str
    ) -> List[dict]:
        """
        Extract phoneme-level timestamps for precise lip-sync
        
        Args:
            audio_path: Audio file path
            transcript: Text transcript
            
        Returns:
            List of phoneme timing information
        """
        # TODO: Implement phoneme extraction using:
        # - Montreal Forced Aligner
        # - Wav2Vec2 + CTC alignment
        # - Whisper with word-level timestamps
        
        # Placeholder: Word-level timestamps
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / sr
        
        words = transcript.split()
        word_duration = duration / len(words) if words else 0
        
        timestamps = []
        for i, word in enumerate(words):
            timestamps.append({
                "word": word,
                "start": i * word_duration,
                "end": (i + 1) * word_duration,
                "confidence": 1.0
            })
        
        return timestamps
    
    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        ensure_sync: bool = True
    ) -> str:
        """
        Merge audio and video with optional alignment
        
        Args:
            video_path: Input video file
            audio_path: Input audio file
            output_path: Output video file with audio
            ensure_sync: Automatically align if needed
            
        Returns:
            Output file path
        """
        if ensure_sync:
            # Validate sync first
            validation = self.validate_sync(audio_path, video_path)
            
            if not validation["is_synced"]:
                print(f"⚠️  Sync issue detected: {validation['difference_ms']:.1f}ms difference")
                print("🔧 Auto-aligning audio...")
                
                # Align audio to video
                aligned_audio = str(Path(output_path).parent / "aligned_audio.wav")
                audio_path, align_info = self.align_audio_to_video(
                    audio_path, video_path, aligned_audio, method="stretch"
                )
                print(f"✅ Alignment complete: {align_info['method']}")
        
        # Merge using FFmpeg
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            "-shortest",  # Use shortest stream duration
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")
        
        return frame_count / fps
    
    def batch_align(
        self,
        audio_video_pairs: List[Tuple[str, str]],
        output_dir: str,
        method: str = "stretch"
    ) -> List[dict]:
        """
        Batch process multiple audio-video pairs
        
        Args:
            audio_video_pairs: List of (audio_path, video_path) tuples
            output_dir: Output directory
            method: Alignment method
            
        Returns:
            List of processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, (audio_path, video_path) in enumerate(audio_video_pairs):
            print(f"Processing {i+1}/{len(audio_video_pairs)}...")
            
            output_audio = str(output_dir / f"aligned_audio_{i:04d}.wav")
            output_video = str(output_dir / f"synced_video_{i:04d}.mp4")
            
            try:
                # Align audio
                aligned_audio, align_info = self.align_audio_to_video(
                    audio_path, video_path, output_audio, method
                )
                
                # Merge
                final_video = self.merge_audio_video(
                    video_path, aligned_audio, output_video, ensure_sync=False
                )
                
                results.append({
                    "success": True,
                    "index": i,
                    "output_video": final_video,
                    "alignment_info": align_info
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "index": i,
                    "error": str(e)
                })
        
        return results


# Example usage
if __name__ == "__main__":
    aligner = AudioVideoAligner()
    
    # Validate sync
    # validation = aligner.validate_sync(
    #     audio_path="audio.wav",
    #     video_path="video.mp4"
    # )
    # print("Sync validation:", validation)
    
    # Align and merge
    # result = aligner.merge_audio_video(
    #     video_path="input_video.mp4",
    #     audio_path="generated_audio.wav",
    #     output_path="output/synced_video.mp4",
    #     ensure_sync=True
    # )
    # print("Output:", result)
    
    print("Audio-Video Aligner ready")
