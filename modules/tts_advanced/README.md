# Advanced TTS Module

Complete Text-to-Speech solution with multi-engine support, voice cloning, and lip-sync integration.

## 🎯 Features

### TTS Engines
- ✅ **Edge-TTS** - Ready to use (100+ voices, free)
- 🔧 **CosyVoice** - Advanced voice cloning (Chinese优化)
- 🔧 **GPT-SoVITS** - Few-shot voice cloning (1-5 sec reference)

### Lip-Sync & Alignment
- 👄 **Wav2Lip** - Audio-driven lip-sync
- 🎬 **Audio-Video Alignment** - Automatic sync correction
- ⏱️ **Frame-level Precision** - <50ms accuracy

### Real-time Features (Future)
- 🎙️ **Voice Activity Detection** - Real-time speech detection
- 💬 **Conversational AI** - Live voice chat assistant
- 🎭 **Avatar Animation** - Real-time facial animation

## 📦 Installation

### Quick Start (Edge-TTS only)
```bash
pip install -r backend/requirements.txt
```

### Full Installation

**1. Edge-TTS (Already included)**
```bash
# No additional steps needed
```

**2. CosyVoice (Optional)**
```bash
cd modules/tts_advanced
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
# Download models
```

**3. GPT-SoVITS (Optional)**
```bash
cd modules/tts_advanced
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
# Download pretrained models
```

**4. Wav2Lip (Optional)**
```bash
cd modules/tts_advanced
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
pip install -r requirements.txt
# Download model checkpoint
```

**5. FFmpeg (Required for video processing)**
```bash
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

## 🚀 Quick Start

### 1. Simple TTS (Edge-TTS)

```python
from modules.tts_advanced import EdgeTTSEngine

# Initialize
tts = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural")

# Synthesize
audio_file = tts.synthesize(
    text="你好，这是一个语音合成测试。",
    output_path="output/test.mp3"
)
print(f"Audio saved: {audio_file}")
```

### 2. Unified TTS Manager

```python
from modules.tts_advanced import TTSManager, TTSEngineType

# Initialize with multiple engines
manager = TTSManager(
    default_engine=TTSEngineType.EDGE_TTS,
    enable_lip_sync=False
)

# Synthesize with automatic fallback
result = manager.synthesize(
    text="Hello, world!",
    output_path="output/hello.mp3",
    voice="en-US-JennyNeural"
)

if result["success"]:
    print(f"Audio generated: {result['audio_path']}")
    print(f"Engine used: {result['engine']}")
```

### 3. Voice Cloning (CosyVoice/GPT-SoVITS)

```python
# After installing CosyVoice or GPT-SoVITS

manager = TTSManager(default_engine=TTSEngineType.COSY_VOICE)

result = manager.synthesize(
    text="这是克隆声音测试。",
    output_path="output/cloned.wav",
    reference_audio="reference.wav",  # 3-10 seconds
    reference_text="参考音频的文本内容"
)
```

### 4. TTS + Lip-Sync Pipeline

```python
# Enable lip-sync
manager = TTSManager(
    default_engine=TTSEngineType.EDGE_TTS,
    enable_lip_sync=True
)

# Generate speech + lip-synced video
result = manager.synthesize_with_lip_sync(
    text="你好，欢迎来到AI多媒体平台！",
    face_video="input/face.mp4",  # or image
    output_video="output/lip_synced.mp4"
)

if result["success"]:
    print(f"Video: {result['video_path']}")
    print(f"Audio: {result['audio_path']}")
```

### 5. Audio-Video Alignment

```python
from modules.tts_advanced.audio_video_aligner import AudioVideoAligner

aligner = AudioVideoAligner()

# Validate sync
validation = aligner.validate_sync(
    audio_path="audio.wav",
    video_path="video.mp4"
)
print(f"Sync quality: {validation['quality']}")
print(f"Difference: {validation['difference_ms']:.1f}ms")

# Auto-align and merge
output = aligner.merge_audio_video(
    video_path="video.mp4",
    audio_path="audio.wav",
    output_path="output/synced.mp4",
    ensure_sync=True  # Auto-correct misalignment
)
```

### 6. Batch Processing

```python
# Batch TTS + Lip-sync
texts = [
    "第一句话",
    "第二句话",
    "第三句话"
]

results = manager.batch_synthesize_with_lip_sync(
    texts=texts,
    face_video="input/avatar.mp4",
    output_dir="output/batch"
)

for i, result in enumerate(results):
    if result["success"]:
        print(f"{i+1}. {result['video_path']}")
```

## 🎨 Voice Selection

### Edge-TTS Voices

```python
from modules.tts_advanced import EdgeTTSEngine

# List all available voices
voices = EdgeTTSEngine.list_voices()
for voice in voices[:10]:
    print(f"{voice['short_name']}: {voice['language']}")

# Popular voices
VOICES = {
    # Chinese
    "zh-CN-XiaoxiaoNeural": "Chinese Female (Natural)",
    "zh-CN-YunxiNeural": "Chinese Male (Natural)",
    
    # English
    "en-US-JennyNeural": "US Female",
    "en-US-GuyNeural": "US Male",
    "en-GB-SoniaNeural": "UK Female",
    
    # Japanese
    "ja-JP-NanamiNeural": "Japanese Female",
}
```

### Voice Parameters

```python
tts = EdgeTTSEngine()

# Adjust voice characteristics
result = tts.synthesize(
    text="可调节的语音参数",
    output_path="output/custom_voice.mp3",
    rate="+20%",      # Speech speed: -50% to +100%
    volume="+10%",    # Volume: -50% to +50%
    pitch="+5Hz"      # Pitch adjustment
)
```

## 🔧 Advanced Usage

### Custom Engine Configuration

```python
# CosyVoice with custom model
from modules.tts_advanced.cosyvoice_engine import CosyVoiceEngine

cosy = CosyVoiceEngine(model_path="custom_models/CosyVoice-300M")

audio = cosy.clone_voice(
    text="自定义声音克隆",
    reference_audio="my_voice.wav",
    reference_text="参考文本",
    output_path="output/my_cloned_voice.wav"
)
```

### Lip-Sync Quality Control

```python
from modules.tts_advanced.lip_sync import LipSyncEngine

lip_sync = LipSyncEngine()

video = lip_sync.generate_lip_sync_video(
    face_video="face.mp4",
    audio_file="speech.wav",
    output_path="output/synced.mp4",
    face_detect_batch_size=16,  # Adjust for GPU memory
    wav2lip_batch_size=128,
    resize_factor=1,  # 1 = original size, 2 = 2x upscale
    fps=25  # Output FPS
)
```

### Real-time Processing (Future)

```python
from modules.tts_advanced.lip_sync import RealTimeLipSync

# For live streaming / video calls
rt_sync = RealTimeLipSync()

# Process audio frames
audio_chunk = np.array([...])  # Audio buffer
mouth_params = rt_sync.process_audio_frame(audio_chunk)

# Apply to avatar
# avatar.set_mouth_open(mouth_params["mouth_open"])
```

## 📊 Performance

| Operation | Speed | GPU | CPU |
|-----------|-------|-----|-----|
| Edge-TTS (10s audio) | ~2s | N/A | ✅ |
| CosyVoice (10s audio) | ~5s | ✅ | ⚠️ Slow |
| GPT-SoVITS (10s audio) | ~3s | ✅ | ⚠️ Slow |
| Wav2Lip (10s video) | ~15s | ✅ | ❌ Very slow |
| Audio alignment | <1s | N/A | ✅ |

## 🎯 Use Cases

### 1. Video Dubbing
```python
# Generate dubbed video with lip-sync
manager = TTSManager(enable_lip_sync=True)

result = manager.synthesize_with_lip_sync(
    text="配音文本",
    face_video="original_video.mp4",
    output_video="dubbed_video.mp4",
    engine=TTSEngineType.COSY_VOICE,
    reference_audio="target_voice.wav"
)
```

### 2. Digital Avatar
```python
# Create talking avatar
avatar_video = "avatar_idle.mp4"

texts = [
    "欢迎光临！",
    "我是你的AI助手。",
    "有什么可以帮您的吗？"
]

for i, text in enumerate(texts):
    manager.synthesize_with_lip_sync(
        text=text,
        face_video=avatar_video,
        output_video=f"avatar_speak_{i}.mp4"
    )
```

### 3. Multilingual Content
```python
# Generate multilingual videos
languages = {
    "zh": ("你好，世界！", "zh-CN-XiaoxiaoNeural"),
    "en": ("Hello, world!", "en-US-JennyNeural"),
    "ja": ("こんにちは、世界！", "ja-JP-NanamiNeural")
}

for lang, (text, voice) in languages.items():
    manager.synthesize_with_lip_sync(
        text=text,
        voice=voice,
        face_video="neutral_face.mp4",
        output_video=f"greeting_{lang}.mp4"
    )
```

## ⚠️ Common Issues

### 1. Audio-Video Sync Problems

**Problem**: Lip movements don't match audio

**Solutions**:
- Use `AudioVideoAligner` before lip-sync
- Ensure consistent frame rate (25fps recommended)
- Check audio sample rate (16kHz recommended)

```python
# Auto-fix sync issues
aligner = AudioVideoAligner()
aligned_audio, info = aligner.align_audio_to_video(
    audio_path="speech.wav",
    video_path="video.mp4",
    output_audio="aligned.wav",
    method="stretch"  # or "trim", "pad"
)
```

### 2. Voice Quality Issues

**Problem**: Generated voice sounds robotic

**Solutions**:
- Use voice cloning (CosyVoice/GPT-SoVITS)
- Adjust speech rate and pitch
- Use higher quality reference audio (>3 seconds, clear)

### 3. Slow Processing

**Problem**: Lip-sync takes too long

**Solutions**:
- Use GPU acceleration
- Reduce video resolution
- Process in smaller batches
- Use Edge-TTS for faster TTS

## 🔗 References

- [Edge-TTS Documentation](https://github.com/rany2/edge-tts)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Wav2Lip Paper](https://arxiv.org/abs/2008.10010)

## 📝 License

See project root LICENSE file.
