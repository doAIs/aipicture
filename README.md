# AI 图片生成项目

一个基于 Stable Diffusion 的 AI 图片生成项目，支持文本生成图片（Text-to-Image）和图片生成图片（Image-to-Image）功能。项目采用由简入深的设计，从最基础的示例到高级参数控制，帮助用户逐步学习和使用 AI 图片生成技术。

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

- **文本生成图片（Text-to-Image）**：根据文字描述生成图片
- **图片生成图片（Image-to-Image）**：基于输入图片和文字描述生成新图片
- **由简入深**：从最基础的示例到高级参数控制
- **参数可调**：支持多种参数精细控制生成效果
- **易于使用**：清晰的代码结构和详细的注释
- **专业实现**：使用 Hugging Face Diffusers 库，稳定可靠

## 🔧 环境要求

- Python 3.8+
- CUDA 11.8+（可选，用于 GPU 加速，CPU 也可运行但速度较慢）
- 至少 8GB 内存（推荐 16GB+）
- 至少 10GB 可用磁盘空间（用于下载模型）

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

#### 基础示例 1：文本生成图片

最简单的文本生成图片示例：

```bash
python 01_basic_text_to_image.py
```

这将根据预设的提示词生成一张图片。

#### 基础示例 2：图片生成图片

根据输入图片生成新图片：

```bash
# 首先准备一张输入图片到 examples 目录
python 02_basic_image_to_image.py
```

#### 进阶示例

查看高级参数控制示例：

```bash
python 03_advanced_text_to_image.py
python 04_advanced_image_to_image.py
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
│
├── 01_basic_text_to_image.py      # 基础示例1：文本生成图片
├── 02_basic_image_to_image.py     # 基础示例2：图片生成图片
├── 03_advanced_text_to_image.py   # 进阶示例1：文本生成图片（带参数）
├── 04_advanced_image_to_image.py  # 进阶示例2：图片生成图片（带参数）
├── 05_comprehensive_example.py    # 综合示例：完整功能演示
│
├── outputs/                       # 生成的图片输出目录
│   ├── basic_text_to_image/       # 基础文本生成图片输出
│   ├── basic_image_to_image/      # 基础图片生成图片输出
│   ├── advanced_text_to_image/    # 进阶文本生成图片输出
│   ├── advanced_image_to_image/   # 进阶图片生成图片输出
│   └── comprehensive/             # 综合示例输出
│
└── examples/                      # 示例图片目录
    ├── README.md                  # 示例图片说明
    └── [你的输入图片]
```

## 💡 使用示例

### 示例 1：基础文本生成图片

```python
from 01_basic_text_to_image import generate_image_from_text

# 生成一张简单的图片
prompt = "a beautiful sunset over the ocean, peaceful, serene"
generate_image_from_text(prompt, "sunset_ocean")
```

### 示例 2：基础图片生成图片

```python
from 02_basic_image_to_image import generate_image_from_image

# 将图片转换为油画风格
generate_image_from_image(
    image_path="examples/input.jpg",
    prompt="oil painting style, artistic, detailed",
    strength=0.7,
    output_name="oil_painting"
)
```

### 示例 3：高级文本生成图片（带参数控制）

```python
from 03_advanced_text_to_image import AdvancedTextToImage

generator = AdvancedTextToImage()

# 使用负面提示词和自定义参数
generator.generate(
    prompt="a beautiful landscape, mountains, lake, peaceful",
    negative_prompt="blurry, low quality, distorted, ugly",
    num_inference_steps=60,
    guidance_scale=7.5,
    height=768,
    width=768,
    seed=42,
    output_name="landscape_highres"
)
```

### 示例 4：高级图片生成图片（带参数控制）

```python
from 04_advanced_image_to_image import AdvancedImageToImage

generator = AdvancedImageToImage()

# 精细控制图片转换
generator.generate(
    image_path="examples/input.jpg",
    prompt="anime style, vibrant colors, detailed",
    negative_prompt="distorted, blurry, low quality",
    strength=0.75,  # 修改强度
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    output_name="anime_style"
)
```

## ⚙️ 参数说明

### 文本生成图片参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | str | 必需 | 正面提示词，描述想要生成的内容 |
| `negative_prompt` | str | None | 负面提示词，描述不想要的内容 |
| `num_inference_steps` | int | 50 | 推理步数（20-100），越多质量越好但越慢 |
| `guidance_scale` | float | 7.5 | 引导强度（1-20），越高越遵循提示词 |
| `height` | int | 512 | 图片高度（必须是8的倍数） |
| `width` | int | 512 | 图片宽度（必须是8的倍数） |
| `seed` | int | None | 随机种子，用于复现结果 |

### 图片生成图片参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | str | 必需 | 输入图片路径 |
| `prompt` | str | 必需 | 正面提示词 |
| `negative_prompt` | str | None | 负面提示词 |
| `strength` | float | 0.75 | 修改强度（0.0-1.0）<br>0.0: 几乎不改变原图<br>0.5: 中等改变<br>1.0: 完全重新生成 |
| `num_inference_steps` | int | 50 | 推理步数 |
| `guidance_scale` | float | 7.5 | 引导强度 |
| `seed` | int | None | 随机种子 |

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

4. **组合技巧**：使用逗号分隔多个概念，使用括号调整权重
   - 示例：`"a cat, (cute:1.2), cartoon style, colorful"`

### 参数调优建议

1. **推理步数（num_inference_steps）**
   - 20-30 步：快速预览
   - 50 步：平衡质量和速度（推荐）
   - 80-100 步：最高质量（较慢）

2. **引导强度（guidance_scale）**
   - 1-5：创意性强，但可能偏离提示词
   - 7-9：平衡（推荐）
   - 10-20：严格遵循提示词，但可能过于生硬

3. **修改强度（strength，仅图片生成图片）**
   - 0.3-0.5：轻微修改，保持原图结构
   - 0.6-0.8：中等修改（推荐）
   - 0.9-1.0：大幅修改，接近重新生成

4. **图片尺寸**
   - 512x512：标准尺寸，速度快
   - 768x768：高分辨率，质量更好
   - 1024x1024：超高分辨率（需要更多内存和时间）

### 性能优化

1. **使用 GPU**：如果有 NVIDIA GPU，确保安装了 CUDA 版本的 PyTorch
2. **启用注意力切片**：代码已自动启用，可节省内存
3. **使用 float16**：在 GPU 上使用 float16 可节省内存并加速
4. **批量生成**：如果需要生成多张图片，考虑批量处理

## ❓ 常见问题

### Q1: 模型下载很慢怎么办？

A: 模型首次运行会自动从 Hugging Face 下载。如果下载慢，可以：
- 使用镜像站点
- 手动下载模型到本地
- 使用 VPN 或代理

### Q2: 内存不足怎么办？

A: 
- 使用 CPU 模式（虽然慢但内存占用小）
- 减小图片尺寸（512x512 而不是 768x768）
- 减少推理步数
- 确保已启用注意力切片（代码已自动处理）

### Q3: 生成的图片质量不好？

A:
- 增加推理步数（50-80 步）
- 使用更详细的提示词
- 添加负面提示词排除不想要的内容
- 尝试不同的随机种子

### Q4: 如何复现相同的结果？

A: 使用固定的 `seed` 参数，相同的提示词和参数会生成相同的结果。

### Q5: CPU 运行太慢怎么办？

A: 
- 考虑使用 GPU（NVIDIA GPU + CUDA）
- 减少推理步数
- 使用较小的图片尺寸
- 考虑使用云端 GPU 服务

## 🛠️ 技术栈

- **PyTorch**: 深度学习框架
- **Diffusers**: Hugging Face 的扩散模型库
- **Transformers**: 预训练模型库
- **Pillow**: 图像处理
- **Stable Diffusion**: 使用的生成模型

## 📚 更多文档

- [详细使用指南](USAGE_GUIDE.md) - 包含提示词编写、参数调优、最佳实践等
- [示例图片说明](examples/README.md) - 如何准备和使用输入图片

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请提交 Issue。

## 🎓 学习路径建议

1. **初学者**：
   - 运行 `quick_start.py` 体验所有功能
   - 阅读 `01_basic_text_to_image.py` 和 `02_basic_image_to_image.py` 的代码
   - 尝试修改提示词，观察效果

2. **进阶用户**：
   - 学习 `03_advanced_text_to_image.py` 和 `04_advanced_image_to_image.py`
   - 阅读 `USAGE_GUIDE.md` 了解参数调优
   - 尝试不同的参数组合

3. **高级用户**：
   - 研究 `05_comprehensive_example.py` 的完整工作流
   - 自定义和扩展功能
   - 优化性能和使用体验

---

**注意**：首次运行需要下载模型（约 4-5GB），请确保网络连接稳定并有足够的磁盘空间。
