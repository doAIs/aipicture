"""
Edge TTS Engine - Microsoft Edge Text-to-Speech
Free, high-quality, 100+ voices in multiple languages
"""

import asyncio
import edge_tts
from pathlib import Path
from typing import Optional, List, Dict


class EdgeTTSEngine:
    """Edge TTS synthesis engine"""
    
    # Popular voices
    VOICES = {
        # Chinese
        "zh-CN-XiaoxiaoNeural": "Chinese Female (Xiaoxiao)",
        "zh-CN-YunxiNeural": "Chinese Male (Yunxi)",
        "zh-CN-YunjianNeural": "Chinese Male (Yunjian)",
        "zh-CN-XiaoyiNeural": "Chinese Female (Xiaoyi)",
        
        # English
        "en-US-JennyNeural": "English US Female (Jenny)",
        "en-US-GuyNeural": "English US Male (Guy)",
        "en-GB-SoniaNeural": "English UK Female (Sonia)",
        "en-GB-RyanNeural": "English UK Male (Ryan)",
        
        # Japanese
        "ja-JP-NanamiNeural": "Japanese Female (Nanami)",
        "ja-JP-KeitaNeural": "Japanese Male (Keita)",
    }
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        """
        Initialize Edge TTS engine
        
        Args:
            voice: Voice name (e.g., "zh-CN-XiaoxiaoNeural")
        """
        self.voice = voice
        self.rate = "+0%"  # Speech rate: -50% to +100%
        self.volume = "+0%"  # Volume: -50% to +100%
        self.pitch = "+0Hz"  # Pitch adjustment
    
    async def synthesize_async(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None
    ) -> str:
        """
        Synthesize speech asynchronously
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            voice: Override default voice
            rate: Speech rate (e.g., "+20%", "-10%")
            volume: Volume (e.g., "+10%", "-20%")
            pitch: Pitch (e.g., "+5Hz", "-10Hz")
            
        Returns:
            Path to generated audio file
        """
        voice = voice or self.voice
        rate = rate or self.rate
        volume = volume or self.volume
        pitch = pitch or self.pitch
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create communicate object
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch
        )
        
        # Save to file
        await communicate.save(output_path)
        
        return output_path
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> str:
        """
        Synchronous wrapper for synthesize_async
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            **kwargs: Additional arguments for synthesize_async
            
        Returns:
            Path to generated audio file
        """
        return asyncio.run(self.synthesize_async(text, output_path, **kwargs))
    
    @staticmethod
    async def list_voices_async() -> List[Dict[str, str]]:
        """
        Get list of all available voices
        
        Returns:
            List of voice information dictionaries
        """
        voices = await edge_tts.list_voices()
        return [
            {
                "name": voice["Name"],
                "short_name": voice["ShortName"],
                "gender": voice["Gender"],
                "locale": voice["Locale"],
                "language": voice["FriendlyName"]
            }
            for voice in voices
        ]
    
    @staticmethod
    def list_voices() -> List[Dict[str, str]]:
        """Synchronous wrapper for list_voices_async"""
        return asyncio.run(EdgeTTSEngine.list_voices_async())
    
    def set_voice_parameters(
        self,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None
    ):
        """
        Update voice parameters
        
        Args:
            rate: Speech rate (e.g., "+20%")
            volume: Volume (e.g., "+10%")
            pitch: Pitch (e.g., "+5Hz")
        """
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch


# Example usage
if __name__ == "__main__":
    # Initialize engine
    tts = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural")
    
    # Synthesize speech
    text = "你好，这是一个语音合成测试。Hello, this is a text-to-speech test."
    output_file = "output/edge_tts_test.mp3"
    
    result = tts.synthesize(text, output_file)
    print(f"Audio saved to: {result}")
    
    # List all available voices
    # voices = EdgeTTSEngine.list_voices()
    # for voice in voices[:5]:
    #     print(f"{voice['name']}: {voice['language']}")
