# AI 多媒体生成与识别项目

一个全面的 AI 多媒体处理项目，支持文本生成图片/视频、图片生成图片/视频、视频生成视频，以及图片/视频识别功能。项目采用由简入深的设计，从最基础的示例到高级参数控制，帮助用户逐步学习和使用 AI 多媒体技术。

## 📋 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用示例](#使用示例)
- [参数说明](#参数说明)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)
- [技术栈](#技术栈)

## ✨ 功能特性

### 生成功能
- **文本生成图片（Text-to-Image）**：根据文字描述生成图片
- **文本生成视频（Text-to-Video）**：根据文字描述生成视频
- **图片生成图片（Image-to-Image）**：基于输入图片和文字描述生成新图片
- **图片生成视频（Image-to-Video）**：基于输入图片生成视频
- **视频生成视频（Video-to-Video）**：基于输入视频和文字描述生成新视频

### 识别功能
- **图片识别（Image Recognition）**：图片分类和目标检测
- **视频识别（Video Recognition）**：视频分类和目标检测

### 项目特点
- **由简入深**：从最基础的示例到高级参数控制
- **参数可调**：支持多种参数精细控制生成效果
- **易于使用**：清晰的代码结构和详细的注释
- **专业实现**：使用 Hugging Face Diffusers、Transformers 等库，稳定可靠
- **模块化设计**：按功能模块组织，便于维护和扩展

## 🔧 环境要求

- Python 3.8+
- CUDA 11.8+（可选，用于 GPU 加速，CPU 也可运行但速度较慢）
- 至少 8GB 内存（推荐 16GB+）
- 至少 20GB 可用磁盘空间（用于下载模型）
- 视频处理功能需要更多内存和存储空间

## 📦 安装步骤

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd aipicture
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
# 方式1: 运行验证脚本（推荐）
python verify_project.py

# 方式2: 手动验证
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 🚀 快速开始

### 方式1：使用快速开始脚本（推荐）

最简单的开始方式，交互式菜单引导：

```bash
python quick_start.py
```

这将显示一个菜单，你可以选择要运行的功能。

### 方式2：直接运行示例文件

#### 文本生成图片

```bash
# 基础版本
python modules/text_to_image/01_basic_text_to_image.py

# 进阶版本
python modules/text_to_image/02_advanced_text_to_image.py
```

#### 图片生成图片

```bash
# 基础版本
python modules/image_to_image/01_basic_image_to_image.py

# 进阶版本
python modules/image_to_image/02_advanced_image_to_image.py
```

#### 文本生成视频

```bash
# 基础版本
python modules/text_to_video/01_basic_text_to_video.py

# 进阶版本
python modules/text_to_video/02_advanced_text_to_video.py
```

#### 图片生成视频

```bash
# 基础版本
python modules/image_to_video/01_basic_image_to_video.py

# 进阶版本
python modules/image_to_video/02_advanced_image_to_video.py
```

#### 视频生成视频

```bash
# 基础版本
python modules/video_to_video/01_basic_video_to_video.py

# 进阶版本
python modules/video_to_video/02_advanced_video_to_video.py
```

#### 图片识别

```bash
# 基础版本
python modules/image_recognition/01_basic_image_recognition.py

# 进阶版本
python modules/image_recognition/02_advanced_image_recognition.py
```

#### 视频识别

```bash
# 基础版本
python modules/video_recognition/01_basic_video_recognition.py

# 进阶版本
python modules/video_recognition/02_advanced_video_recognition.py
```

#### 综合示例

运行完整功能演示：

```bash
python 05_comprehensive_example.py
```

## 📁 项目结构

```
aipicture/
├── README.md                      # 项目说明文档（本文件）
├── USAGE_GUIDE.md                 # 详细使用指南
├── requirements.txt               # 项目依赖
├── config.py                      # 配置文件
├── utils.py                       # 工具函数
│
├── quick_start.py                 # 快速开始脚本（推荐新手使用）
├── 05_comprehensive_example.py    # 综合示例：完整功能演示
│
├── modules/                       # 功能模块目录
│   ├── text_to_image/             # 文本生成图片模块
│   │   ├── __init__.py
│   │   ├── 01_basic_text_to_image.py
│   │   └── 02_advanced_text_to_image.py
│   │
│   ├── text_to_video/             # 文本生成视频模块
│   │   ├── __init__.py
│   │   ├── 01_basic_text_to_video.py
│   │   └── 02_advanced_text_to_video.py
│   │
│   ├── image_to_image/            # 图片生成图片模块
│   │   ├── __init__.py
│   │   ├── 01_basic_image_to_image.py
│   │   └── 02_advanced_image_to_image.py
│   │
│   ├── image_to_video/            # 图片生成视频模块
│   │   ├── __init__.py
│   │   ├── 01_basic_image_to_video.py
│   │   └── 02_advanced_image_to_video.py
│   │
│   ├── video_to_video/            # 视频生成视频模块
│   │   ├── __init__.py
│   │   ├── 01_basic_video_to_video.py
│   │   └── 02_advanced_video_to_video.py
│   │
│   ├── image_recognition/         # 图片识别模块
│   │   ├── __init__.py
│   │   ├── 01_basic_image_recognition.py
│   │   └── 02_advanced_image_recognition.py
│   │
│   └── video_recognition/         # 视频识别模块
│       ├── __init__.py
│       ├── 01_basic_video_recognition.py
│       └── 02_advanced_video_recognition.py
│
├── outputs/                       # 输出目录
│   ├── images/                    # 生成的图片输出
│   │   ├── basic_text_to_image/
│   │   ├── advanced_text_to_image/
│   │   ├── basic_image_to_image/
│   │   ├── advanced_image_to_image/
│   │   └── ...
│   └── videos/                    # 生成的视频输出
│       ├── basic_text_to_video/
│       ├── advanced_text_to_video/
│       └── ...
│
└── examples/                      # 示例文件目录
    ├── images/                    # 示例图片
    └── videos/                    # 示例视频
```

## 💡 使用示例

### 示例 1：文本生成图片

```python
from modules.text_to_image import AdvancedTextToImage

generator = AdvancedTextToImage()

# 生成一张图片
generator.generate(
    prompt="a beautiful sunset over the ocean, peaceful, serene",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    output_name="sunset_ocean"
)
```

### 示例 2：文本生成视频

```python
from modules.text_to_video import AdvancedTextToVideo

generator = AdvancedTextToVideo()

# 生成一个视频
generator.generate(
    prompt="a beautiful sunset over the ocean, peaceful, serene",
    num_frames=16,
    fps=8,
    output_name="sunset_video"
)
```

### 示例 3：图片生成图片

```python
from modules.image_to_image import AdvancedImageToImage

generator = AdvancedImageToImage()

# 将图片转换为油画风格
generator.generate(
    image_path="examples/images/input.jpg",
    prompt="oil painting style, artistic, detailed",
    strength=0.7,
    output_name="oil_painting"
)
```

### 示例 4：图片生成视频

```python
from modules.image_to_video import AdvancedImageToVideo

generator = AdvancedImageToVideo()

# 根据图片生成视频
generator.generate(
    image_path="examples/images/input.jpg",
    num_frames=14,
    fps=7,
    motion_bucket_id=127,
    output_name="image_to_video"
)
```

### 示例 5：视频生成视频

```python
from modules.video_to_video import AdvancedVideoToVideo

generator = AdvancedVideoToVideo()

# 将视频转换为动画风格
generator.generate(
    video_path="examples/videos/input.mp4",
    prompt="anime style, vibrant colors, detailed",
    strength=0.75,
    num_frames=30,
    fps=8,
    output_name="anime_video"
)
```

### 示例 6：图片识别

```python
from modules.image_recognition import AdvancedImageRecognition

recognizer = AdvancedImageRecognition()

# 图片分类
results = recognizer.classify("examples/images/test.jpg", top_k=5)
for label, confidence in results:
    print(f"{label}: {confidence:.2f}%")

# 目标检测
detections = recognizer.detect("examples/images/test.jpg", confidence_threshold=0.5)
for det in detections:
    print(f"{det['label']}: {det['confidence']:.2f}%")
```

### 示例 7：视频识别

```python
from modules.video_recognition import AdvancedVideoRecognition

recognizer = AdvancedVideoRecognition()

# 视频分类
results = recognizer.classify("examples/videos/test.mp4", sample_frames=10, top_k=5)
for result in results:
    print(f"{result['label']}: {result['percentage']:.1f}%")

# 视频目标检测
detections = recognizer.detect("examples/videos/test.mp4", sample_frames=10)
for result in detections:
    print(f"{result['label']}: {result['detections']} 次检测")
```

## ⚙️ 参数说明

### 图片生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | 必需 | 正面提示词，描述想要生成的内容 |
| `negative_prompt` | str | None | 负面提示词，描述不想要的内容 |
| `num_inference_steps` | int | 50 | 推理步数（20-100），越多质量越好但越慢 |
| `guidance_scale` | float | 7.5 | 引导强度（1-20），越高越遵循提示词 |
| `height` | int | 512 | 图片高度（必须是8的倍数） |
| `width` | int | 512 | 图片宽度（必须是8的倍数） |
| `seed` | int | None | 随机种子，用于复现结果 |
| `strength` | float | 0.75 | 修改强度（0.0-1.0），仅用于图片/视频生成 |

### 视频生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_frames` | int | 16 | 视频帧数（8-32），越多越流畅但越慢 |
| `fps` | int | 8 | 帧率（4-12），影响播放速度 |
| `motion_bucket_id` | int | 127 | 运动强度（1-255），仅用于图片生成视频 |
| `height` | int | 256 | 视频高度（必须是8的倍数） |
| `width` | int | 256 | 视频宽度（必须是8的倍数） |

### 识别参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | int | 5 | 返回前k个最可能的类别 |
| `confidence_threshold` | float | 0.5 | 置信度阈值（0.0-1.0） |
| `sample_frames` | int | 10 | 采样帧数（仅用于视频识别） |

## 🎯 最佳实践

### 提示词编写技巧

1. **具体描述**：使用具体、详细的描述，而不是抽象概念
   - ✅ 好：`"a majestic lion standing on a rock at sunset, golden hour lighting, photorealistic, detailed fur"`
   - ❌ 差：`"a lion"`

2. **使用风格关键词**：添加艺术风格、质量描述
   - 风格：`"oil painting"`, `"anime style"`, `"cyberpunk"`, `"watercolor"`
   - 质量：`"high quality"`, `"detailed"`, `"4k"`, `"professional"`

3. **负面提示词**：使用负面提示词排除不想要的内容
   - 常见负面词：`"blurry"`, `"low quality"`, `"distorted"`, `"ugly"`, `"deformed"`

### 参数调优建议

1. **推理步数（num_inference_steps）**
   - 20-30 步：快速预览
   - 50 步：平衡质量和速度（推荐）
   - 80-100 步：最高质量（较慢）

2. **引导强度（guidance_scale）**
   - 1-5：创意性强，但可能偏离提示词
   - 7-9：平衡（推荐）
   - 10-20：严格遵循提示词，但可能过于生硬

3. **修改强度（strength，仅图片/视频生成）**
   - 0.3-0.5：轻微修改，保持原图结构
   - 0.6-0.8：中等修改（推荐）
   - 0.9-1.0：大幅修改，接近重新生成

4. **视频帧数（num_frames）**
   - 8-14 帧：快速生成，适合简单动画
   - 16-24 帧：平衡质量和速度（推荐）
   - 25-32 帧：高质量，流畅动画（较慢）

### 性能优化

1. **使用 GPU**：如果有 NVIDIA GPU，确保安装了 CUDA 版本的 PyTorch
2. **启用注意力切片**：代码已自动启用，可节省内存
3. **使用 float16**：在 GPU 上使用 float16 可节省内存并加速
4. **批量处理**：如果需要处理多张图片/视频，考虑批量处理
5. **视频处理**：视频处理非常耗时，建议先处理少量帧测试效果

## ❓ 常见问题

### Q1: 模型下载很慢怎么办？

A: 模型首次运行会自动从 Hugging Face 下载。如果下载慢，可以：
- 使用镜像站点
- 手动下载模型到本地
- 使用 VPN 或代理

### Q2: 内存不足怎么办？

A: 
- 使用 CPU 模式（虽然慢但内存占用小）
- 减小图片/视频尺寸
- 减少推理步数
- 减少视频帧数
- 确保已启用注意力切片（代码已自动处理）

### Q3: 生成的图片/视频质量不好？

A:
- 增加推理步数（50-80 步）
- 使用更详细的提示词
- 添加负面提示词排除不想要的内容
- 尝试不同的随机种子

### Q4: 视频生成太慢怎么办？

A:
- 减少视频帧数（8-14 帧）
- 使用较小的视频尺寸
- 使用 GPU 加速
- 考虑使用更快的模型

### Q5: CPU 运行太慢怎么办？

A: 
- 考虑使用 GPU（NVIDIA GPU + CUDA）
- 减少推理步数
- 使用较小的图片/视频尺寸
- 考虑使用云端 GPU 服务

## 🛠️ 技术栈

- **PyTorch**: 深度学习框架
- **Diffusers**: Hugging Face 的扩散模型库
- **Transformers**: 预训练模型库
- **Pillow**: 图像处理
- **OpenCV**: 视频处理
- **MoviePy**: 视频编辑
- **YOLO**: 目标检测
- **Stable Diffusion**: 图片生成模型
- **Stable Video Diffusion**: 视频生成模型

## 📚 更多文档

- [详细使用指南](USAGE_GUIDE.md) - 包含提示词编写、参数调优、最佳实践等
- [示例文件说明](examples/README.md) - 如何准备和使用输入文件

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请提交 Issue。

## 🎓 学习路径建议

1. **初学者**：
   - 运行 `quick_start.py` 体验所有功能
   - 阅读基础示例代码（01_*.py）
   - 尝试修改提示词，观察效果

2. **进阶用户**：
   - 学习进阶示例代码（02_*.py）
   - 阅读 `USAGE_GUIDE.md` 了解参数调优
   - 尝试不同的参数组合

3. **高级用户**：
   - 研究 `05_comprehensive_example.py` 的完整工作流
   - 自定义和扩展功能
   - 优化性能和使用体验

---

**注意**：首次运行需要下载模型（约 10-20GB），请确保网络连接稳定并有足够的磁盘空间。视频生成功能需要更多时间和资源，请耐心等待。
