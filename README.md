# AI Multimedia Platform

A professional commercial-grade AI multimedia processing platform with a modern Vue 3 frontend, FastAPI backend, real-time WebSocket communication, model training capabilities, and comprehensive documentation.

## ✨ Features

### Generation Capabilities
- **Text to Image**: Generate stunning images from text descriptions using Stable Diffusion
- **Image to Image**: Transform and enhance images with AI-powered editing
- **Text to Video**: Create videos from text descriptions
- **Image to Video**: Animate images into dynamic videos
- **Video to Video**: Transform videos with style transfer and enhancement

### Recognition Capabilities
- **Image Recognition**: Object detection and classification using YOLO
- **Video Recognition**: Real-time video analysis and object tracking
- **Face Recognition**: Face detection, recognition, and database management
- **Live Camera**: Real-time recognition with webcam support

### Audio Processing
- **Speech to Text**: Transcribe audio using Whisper models
- **Text to Speech**: Generate natural speech from text
- **Audio Classification**: Classify audio content

### Training Capabilities
- **Model Training**: Train custom image classifiers and object detectors
- **LLM Fine-tuning**: Fine-tune large language models with LoRA/QLoRA
- **Dataset Management**: Upload and manage training datasets
- **Real-time Progress**: WebSocket-based training progress monitoring

## 🏗️ Architecture

```
aipicture/
├── vue-web/                    # Vue 3 Frontend (NEW)
│   ├── src/
│   │   ├── api/                # API client and WebSocket
│   │   ├── components/         # Reusable UI components
│   │   ├── views/              # Page components
│   │   ├── stores/             # Pinia state management
│   │   └── styles/             # Tech/sci-fi themed styles
│   └── package.json
│
├── backend/                    # FastAPI Backend (NEW)
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── routes/             # REST API endpoints
│   │   └── websocket/          # WebSocket handlers
│   ├── services/               # Business logic
│   └── schemas/                # Pydantic models
│
├── modules/                    # AI Processing Modules
│   ├── text_to_image/
│   ├── image_to_image/
│   ├── text_to_video/
│   ├── image_to_video/
│   ├── video_to_video/
│   ├── image_recognition/
│   ├── video_recognition/
│   ├── face_recognition/       # (NEW)
│   ├── audio/                  # (NEW)
│   ├── training/               # (NEW)
│   └── llm_finetuning/         # (NEW)
│
├── config/                     # Configuration
│   └── modules_config.py
│
├── docs/                       # Documentation
│   ├── API_REFERENCE.md
│   ├── TRAINING_GUIDE.md
│   └── DEPLOYMENT.md
│
└── docker/                     # Docker configuration
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    └── docker-compose.yml
```

## 🔧 Requirements

### System Requirements
- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM (32GB recommended for training)
- 50GB+ disk space (for models)

### Hardware Recommendations
| Task | Minimum | Recommended |
|------|---------|-------------|
| Image Generation | 8GB VRAM | 12GB+ VRAM |
| Video Generation | 12GB VRAM | 24GB+ VRAM |
| LLM Fine-tuning | 16GB VRAM | 24GB+ VRAM |
| Training (LoRA) | 8GB VRAM | 16GB+ VRAM |

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aipicture
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 3. Frontend Setup

```bash
cd vue-web
npm install
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Frontend
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# Models
MODEL_CACHE_DIR=./models
HF_TOKEN=your_huggingface_token  # Optional

# GPU
CUDA_VISIBLE_DEVICES=0
```

## 🚀 Quick Start

### Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd vue-web
npm run dev
```

Access the platform at `http://localhost:5173`

### Production Mode

```bash
# Build frontend
cd vue-web
npm run build

# Run with Docker
docker-compose up -d
```

## 📖 Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Training Guide](docs/TRAINING_GUIDE.md) - How to train and fine-tune models
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Usage Guide](USAGE_GUIDE.md) - Detailed feature usage

## 🎨 UI Theme

The frontend features a modern tech/sci-fi theme with:
- Dark gradient backgrounds (#0a0f1e to #1a1f3e)
- Neon accent colors (cyan, magenta, green)
- Glass-morphism effects
- Animated elements and transitions

## 🔌 API Endpoints

### Generation APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/text-to-image/generate` | POST | Generate image from text |
| `/api/image-to-image/generate` | POST | Transform image |
| `/api/text-to-video/generate` | POST | Generate video from text |
| `/api/face-recognition/detect` | POST | Detect faces in image |
| `/api/audio/transcribe` | POST | Transcribe audio to text |

### Training APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/start` | POST | Start model training |
| `/api/training/status/{task_id}` | GET | Get training status |
| `/api/llm-finetuning/lora/start` | POST | Start LoRA fine-tuning |

### WebSocket Endpoints
| Endpoint | Description |
|----------|-------------|
| `/ws/training` | Real-time training progress |
| `/ws/camera` | Live camera feed |
| `/ws/generation` | Generation progress |

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Vue 3, TypeScript, Vite |
| UI Library | Element Plus |
| State Management | Pinia |
| Backend | FastAPI, Python 3.10+ |
| WebSocket | FastAPI WebSocket |
| AI Framework | PyTorch, HuggingFace |
| Image Generation | Stable Diffusion, Diffusers |
| Video Processing | OpenCV, MoviePy |
| Object Detection | YOLOv8 |
| Face Recognition | face_recognition, dlib |
| Audio Processing | Whisper, TTS |
| LLM Fine-tuning | PEFT (LoRA, QLoRA) |

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Performance Tips

1. **GPU Utilization**: Ensure CUDA is properly configured
2. **Memory Management**: Use attention slicing for large models
3. **Batch Processing**: Process multiple items together when possible
4. **Model Caching**: Models are cached after first download
5. **WebSocket**: Use WebSocket for real-time progress updates

## 🔒 Security Considerations

- API authentication (optional, configurable)
- CORS configuration for frontend
- File upload validation
- Rate limiting (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 📧 Support

For issues and feature requests, please open a GitHub issue.

---

**Note**: First run will download AI models (10-50GB depending on features used). Ensure stable internet connection and sufficient disk space.
