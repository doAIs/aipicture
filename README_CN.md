# AI 多媒体平台

一个专业的商用级 AI 多媒体处理平台，具有现代化的 Vue 3 前端、FastAPI 后端、实时 WebSocket 通信、模型训练功能和完整的文档。

[English](README.md) | **中文**

## ✨ 功能特性

### 生成功能
- **文生图 (Text to Image)**: 使用 Stable Diffusion 从文本描述生成精美图像
- **图生图 (Image to Image)**: 使用 AI 技术转换和增强图像
- **文生视频 (Text to Video)**: 从文本描述创建视频
- **图生视频 (Image to Video)**: 将静态图像转换为动态视频
- **视频转视频 (Video to Video)**: 对视频进行风格迁移和增强

### 识别功能
- **图像识别**: 使用 YOLO 进行目标检测和分类
- **视频识别**: 实时视频分析和目标跟踪
- **人脸识别**: 人脸检测、识别和数据库管理
- **实时摄像头**: 支持摄像头实时识别

### 音频处理
- **语音转文字**: 使用 Whisper 模型进行语音转录
- **文字转语音**: 从文本生成自然语音
- **音频分类**: 对音频内容进行分类

### 训练功能
- **模型训练**: 训练自定义图像分类器和目标检测器
- **LLM 微调**: 使用 LoRA/QLoRA 微调大型语言模型
- **数据集管理**: 上传和管理训练数据集
- **实时进度**: 基于 WebSocket 的训练进度监控

## 🏗️ 项目架构

```
aipicture/
├── vue-web/                    # Vue 3 前端（新增）
│   ├── src/
│   │   ├── api/                # API 客户端和 WebSocket
│   │   ├── components/         # 可复用 UI 组件
│   │   ├── views/              # 页面组件
│   │   ├── stores/             # Pinia 状态管理
│   │   └── styles/             # 科技/赛博风格样式
│   └── package.json
│
├── backend/                    # FastAPI 后端（新增）
│   ├── main.py                 # FastAPI 应用入口
│   ├── api/
│   │   ├── routes/             # REST API 端点
│   │   └── websocket/          # WebSocket 处理器
│   ├── services/               # 业务逻辑层
│   └── schemas/                # Pydantic 模型
│
├── modules/                    # AI 处理模块
│   ├── text_to_image/          # 文生图
│   ├── image_to_image/         # 图生图
│   ├── text_to_video/          # 文生视频
│   ├── image_to_video/         # 图生视频
│   ├── video_to_video/         # 视频转视频
│   ├── image_recognition/      # 图像识别
│   ├── video_recognition/      # 视频识别
│   ├── face_recognition/       # 人脸识别（新增）
│   ├── audio/                  # 音频处理（新增）
│   ├── training/               # 模型训练（新增）
│   └── llm_finetuning/         # LLM 微调（新增）
│
├── config/                     # 配置文件
│   └── modules_config.py
│
├── docs/                       # 文档
│   ├── API_REFERENCE.md
│   ├── TRAINING_GUIDE.md
│   └── DEPLOYMENT.md
│
└── docker/                     # Docker 配置
    ├── Dockerfile.backend
    ├── Dockerfile.frontend
    └── docker-compose.yml
```

## 🔧 系统要求

### 基本要求
- Python 3.10+
- Node.js 18+
- CUDA 11.8+（可选，用于 GPU 加速）
- 16GB+ 内存（训练建议 32GB）
- 50GB+ 磁盘空间（用于存储模型）

### 硬件配置建议
| 任务 | 最低配置 | 推荐配置 |
|------|---------|----------|
| 图像生成 | 8GB 显存 | 12GB+ 显存 |
| 视频生成 | 12GB 显存 | 24GB+ 显存 |
| LLM 微调 | 16GB 显存 | 24GB+ 显存 |
| 训练 (LoRA) | 8GB 显存 | 16GB+ 显存 |

## 📦 安装步骤

### 1. 克隆仓库

```bash
git clone <repository-url>
cd aipicture
```

### 2. 后端配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 3. 前端配置

```bash
cd vue-web
npm install
```

### 4. 环境变量配置

在项目根目录创建 `.env` 文件：

```env
# 后端配置
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# 前端配置
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000

# 模型配置
MODEL_CACHE_DIR=./models
HF_TOKEN=your_huggingface_token  # 可选

# GPU 配置
CUDA_VISIBLE_DEVICES=0
```

## 🚀 快速开始

### 开发模式

**终端 1 - 启动后端：**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**终端 2 - 启动前端：**
```bash
cd vue-web
npm run dev
```

访问地址：`http://localhost:5173`

### 使用启动脚本（推荐）

**Windows：**
```batch
# 启动全部服务（开发模式）
scripts\start_all.bat

# 或者单独启动
scripts\start_backend.bat --dev
scripts\start_frontend.bat
```

**Linux/Mac：**
```bash
# 启动全部服务
./scripts/start_all.sh

# 或者单独启动
./scripts/start_backend.sh --dev
./scripts/start_frontend.sh
```

### 生产模式

```bash
# 构建前端
cd vue-web
npm run build

# 使用 Docker 运行
docker-compose up -d
```

## 📖 文档

- [API 参考文档](docs/API_REFERENCE.md) - 完整的 API 文档
- [训练指南](docs/TRAINING_GUIDE.md) - 如何训练和微调模型
- [部署指南](docs/DEPLOYMENT.md) - 生产环境部署说明
- [使用指南](USAGE_GUIDE.md) - 详细功能使用说明

## 🎨 界面主题

前端采用现代科技/赛博朋克风格主题：
- 深色渐变背景 (#0a0f1e 到 #1a1f3e)
- 霓虹色调 (青色、品红色、绿色)
- 毛玻璃效果
- 动画元素和过渡效果

## 🔌 API 接口

### 生成类接口
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/text-to-image/generate` | POST | 从文本生成图像 |
| `/api/image-to-image/generate` | POST | 图像转换 |
| `/api/text-to-video/generate` | POST | 从文本生成视频 |
| `/api/face-recognition/detect` | POST | 检测图像中的人脸 |
| `/api/audio/transcribe` | POST | 语音转文字 |

### 训练类接口
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/training/start` | POST | 开始模型训练 |
| `/api/training/status/{task_id}` | GET | 获取训练状态 |
| `/api/llm-finetuning/lora/start` | POST | 开始 LoRA 微调 |

### WebSocket 接口
| 端点 | 描述 |
|------|------|
| `/ws/training` | 实时训练进度 |
| `/ws/camera` | 实时摄像头画面 |
| `/ws/generation` | 生成进度 |

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Vue 3, TypeScript, Vite |
| UI 库 | Element Plus |
| 状态管理 | Pinia |
| 后端 | FastAPI, Python 3.10+ |
| WebSocket | FastAPI WebSocket |
| AI 框架 | PyTorch, HuggingFace |
| 图像生成 | Stable Diffusion, Diffusers |
| 视频处理 | OpenCV, MoviePy |
| 目标检测 | YOLOv8 |
| 人脸识别 | face_recognition, dlib |
| 音频处理 | Whisper, TTS |
| LLM 微调 | PEFT (LoRA, QLoRA) |

## 🐳 Docker 部署

```bash
# 构建并运行所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 📊 性能优化建议

1. **GPU 利用率**: 确保 CUDA 正确配置
2. **内存管理**: 对大模型使用注意力切片 (attention slicing)
3. **批量处理**: 尽可能批量处理多个任务
4. **模型缓存**: 模型在首次下载后会被缓存
5. **WebSocket**: 使用 WebSocket 获取实时进度更新

## 🔒 安全考虑

- API 认证（可选，可配置）
- 前端 CORS 配置
- 文件上传验证
- 速率限制（可配置）

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📝 许可证

本项目采用 MIT 许可证。

## 📧 支持

如有问题或功能需求，请在 GitHub 上提交 Issue。

---

**注意**: 首次运行时会下载 AI 模型（根据使用的功能，大小为 10-50GB）。请确保网络连接稳定且有足够的磁盘空间。
