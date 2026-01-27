# TTS Advanced Module - Implementation Guide

Complete guide for implementing multi-engine TTS with lip-sync and audio-video alignment.

## 📋 Overview

This module provides a production-ready TTS solution with:
- **3 TTS Engines**: Edge-TTS, CosyVoice, GPT-SoVITS
- **Voice Cloning**: Zero-shot and few-shot capabilities
- **Lip-Sync**: Wav2Lip integration for perfect audio-video sync
- **Auto-Alignment**: Automatic audio-video synchronization correction
- **API Ready**: Complete FastAPI endpoints

## 🎯 Architecture

```
tts_advanced/
├── edge_tts_engine.py          # ✅ Edge-TTS (ready)
├── cosyvoice_engine.py         # 🔧 CosyVoice (install)
├── gpt_sovits_engine.py        # 🔧 GPT-SoVITS (install)
├── lip_sync.py                 # 🔧 Wav2Lip (install)
├── audio_video_aligner.py      # ✅ Alignment (ready)
└── tts_manager.py              # ✅ Unified manager (ready)
```

## 🚀 Phase 1: Edge-TTS (Current ✅)

**Status**: Ready to use immediately

**Features**:
- 100+ voices in multiple languages
- High quality, natural-sounding speech
- Free, no API keys required
- Fast processing (<2s for 10s audio)

**Usage**:
```python
from modules.tts_advanced import EdgeTTSEngine

tts = EdgeTTSEngine(voice="zh-CN-XiaoxiaoNeural")
audio = tts.synthesize("你好，世界！", "output.mp3")
```

**API Endpoint**:
```bash
POST /api/tts-advanced/synthesize
{
  "text": "你好，世界！",
  "voice": "zh-CN-XiaoxiaoNeural",
  "rate": "+0%"
}
```

## 🔧 Phase 2: Voice Cloning (2-4 weeks)

### Option A: CosyVoice (Recommended for Chinese)

**Installation**:
```bash
cd modules/tts_advanced
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt

# Download models
# Model: CosyVoice-300M (~600MB)
```

**Enable in code**:
```python
# In modules/tts_advanced/__init__.py
from .cosyvoice_engine import CosyVoiceEngine

# In modules/tts_advanced/tts_manager.py
from .cosyvoice_engine import CosyVoiceEngine
self.engines[TTSEngineType.COSY_VOICE] = CosyVoiceEngine()
```

**Usage**:
```python
manager = TTSManager(default_engine=TTSEngineType.COSY_VOICE)

result = manager.synthesize(
    text="克隆声音测试",
    output_path="cloned.wav",
    reference_audio="voice_sample.wav",  # 3-10 seconds
    reference_text="声音样本的文本"
)
```

### Option B: GPT-SoVITS (Faster inference)

**Installation**:
```bash
cd modules/tts_advanced
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt

# Download pretrained models
# GPT: s1bert25hz-2kh-longer.ckpt
# SoVITS: s2G488k.pth
```

**Enable in code**:
```python
# Similar to CosyVoice
from .gpt_sovits_engine import GPTSoVITSEngine
```

**Usage**:
```python
result = manager.synthesize(
    text="快速克隆测试",
    output_path="cloned.wav",
    engine=TTSEngineType.GPT_SOVITS,
    reference_audio="voice.wav",  # 1-5 seconds
    reference_text="样本文本"
)
```

## 👄 Phase 3: Lip-Sync (4-6 weeks)

### Wav2Lip Installation

**Step 1: Clone repository**
```bash
cd modules/tts_advanced
git clone https://github.com/Rudrabha/Wav2Lip.git
cd Wav2Lip
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Download models**
```bash
# Download wav2lip_gan.pth model
# Place in: modules/tts_advanced/Wav2Lip/checkpoints/
```

**Step 4: Enable in code**
```python
# In modules/tts_advanced/lip_sync.py
# Uncomment Wav2Lip imports and implementation
```

### Usage

**Simple lip-sync**:
```python
from modules.tts_advanced import TTSManager

manager = TTSManager(enable_lip_sync=True)

result = manager.synthesize_with_lip_sync(
    text="口型对齐测试",
    face_video="face.mp4",
    output_video="synced.mp4"
)
```

**API Endpoint**:
```bash
POST /api/tts-advanced/synthesize-with-lipsync
- text: "口型对齐测试"
- face_video: [file upload]
- voice: "zh-CN-XiaoxiaoNeural"
```

## 🎬 Phase 4: Audio-Video Alignment (Current ✅)

**Status**: Ready to use

**Features**:
- Automatic duration matching
- Time-stretch without pitch change
- Validation (<100ms tolerance)
- Multiple alignment methods

**Usage**:
```python
from modules.tts_advanced.audio_video_aligner import AudioVideoAligner

aligner = AudioVideoAligner()

# Validate sync
validation = aligner.validate_sync("audio.wav", "video.mp4")
print(f"Quality: {validation['quality']}")

# Auto-align and merge
output = aligner.merge_audio_video(
    video_path="video.mp4",
    audio_path="audio.wav",
    output_path="synced.mp4",
    ensure_sync=True
)
```

**API Endpoint**:
```bash
POST /api/tts-advanced/align-audio-video
- audio: [file upload]
- video: [file upload]
- method: "stretch"  # or "trim", "pad"
```

## 📊 Performance Benchmarks

| Operation | GPU | CPU | Notes |
|-----------|-----|-----|-------|
| Edge-TTS (10s) | N/A | 2s | Pure online |
| CosyVoice (10s) | 5s | 30s | GPU recommended |
| GPT-SoVITS (10s) | 3s | 25s | Fastest cloning |
| Wav2Lip (10s video) | 15s | 5min | GPU required |
| Audio alignment | N/A | <1s | Very fast |

**Hardware Recommendations**:
- **Development**: CPU + 16GB RAM
- **Production**: GPU (8GB+ VRAM) + 32GB RAM
- **Lip-sync**: GPU (12GB+ VRAM) required

## 🔗 Integration with Backend

### 1. Register routes in main.py

```python
# backend/main.py
from backend.api.routes import tts_advanced

app.include_router(tts_advanced.router, prefix="/api")
```

### 2. Frontend integration

```typescript
// vue-web/src/api/tts.ts
export async function synthesizeSpeech(params: {
  text: string
  voice?: string
  engine?: string
}) {
  return request.post('/tts-advanced/synthesize', params)
}

export async function synthesizeWithLipSync(params: {
  text: string
  faceVideo: File
  voice?: string
}) {
  const formData = new FormData()
  formData.append('text', params.text)
  formData.append('face_video', params.faceVideo)
  formData.append('voice', params.voice || 'zh-CN-XiaoxiaoNeural')
  
  return request.post('/tts-advanced/synthesize-with-lipsync', formData)
}
```

## 🎯 Production Checklist

### Phase 1 (Current) ✅
- [x] Edge-TTS engine
- [x] TTSManager
- [x] Audio-video aligner
- [x] API endpoints
- [x] Basic testing

### Phase 2 (Weeks 2-4)
- [ ] Install CosyVoice or GPT-SoVITS
- [ ] Enable in TTSManager
- [ ] Test voice cloning
- [ ] Add reference audio management
- [ ] API testing with voice cloning

### Phase 3 (Weeks 4-6)
- [ ] Install Wav2Lip
- [ ] Integrate lip-sync pipeline
- [ ] Test with various videos
- [ ] Optimize for GPU
- [ ] Handle edge cases (multiple faces, etc.)

### Phase 4 (Weeks 6-8)
- [ ] Batch processing optimization
- [ ] Queue system for long tasks
- [ ] WebSocket progress updates
- [ ] Error recovery mechanisms

### Phase 5 (Weeks 8-12) - Real-time Features
- [ ] Real-time VAD integration
- [ ] Live streaming support
- [ ] LLM integration for conversational AI
- [ ] Avatar animation

## 🐛 Troubleshooting

### Problem 1: Lip-sync audio-video mismatch

**Symptoms**: Mouth movements don't match audio

**Solutions**:
1. Use `AudioVideoAligner` before lip-sync
2. Check video FPS (25fps recommended)
3. Verify audio sample rate (16kHz)
4. Adjust `wav2lip_batch_size`

```python
# Pre-align before lip-sync
aligner = AudioVideoAligner()
aligned_audio, info = aligner.align_audio_to_video(
    audio_path="speech.wav",
    video_path="face.mp4",
    output_audio="aligned.wav"
)

# Then run lip-sync with aligned audio
```

### Problem 2: CosyVoice/GPT-SoVITS import errors

**Symptoms**: `ImportError` when loading engines

**Solutions**:
1. Verify installation in correct directory
2. Check Python environment
3. Install missing dependencies
4. Keep engines commented out until ready

### Problem 3: Slow processing

**Symptoms**: Generation takes too long

**Solutions**:
1. Enable GPU acceleration
2. Reduce batch sizes
3. Use Edge-TTS for faster TTS
4. Process smaller video segments
5. Use async/queue for long tasks

## 📚 Additional Resources

- [Edge-TTS Voices List](https://github.com/rany2/edge-tts#usage)
- [CosyVoice Demo](https://fun-audio-llm.github.io)
- [GPT-SoVITS Usage](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/README.md)
- [Wav2Lip Paper](https://arxiv.org/abs/2008.10010)

## 🎓 Training Custom Models (Advanced)

For enterprise deployments, you can train custom models:

### Fine-tune CosyVoice
```bash
# Prepare dataset
# Train on your own voices
# See CosyVoice documentation
```

### Train GPT-SoVITS
```bash
# Collect voice samples (1-10 min per speaker)
# Fine-tune on custom dataset
# See GPT-SoVITS training guide
```

## 📝 Next Steps

1. **Immediate**: Test Edge-TTS with current API
2. **Week 1-2**: Choose and install voice cloning engine
3. **Week 3-4**: Integrate voice cloning into workflow
4. **Week 5-6**: Add Wav2Lip for lip-sync
5. **Week 7+**: Optimize and add real-time features

---

**Questions?** Check the main README.md or create an issue.
