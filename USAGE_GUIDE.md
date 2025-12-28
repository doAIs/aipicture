# 使用指南

本文档提供详细的使用指南和最佳实践，帮助你更好地使用 AI 图片生成项目。

## 📚 目录

- [快速入门](#快速入门)
- [提示词编写指南](#提示词编写指南)
- [参数调优指南](#参数调优指南)
- [实际应用场景](#实际应用场景)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

## 🚀 快速入门

### 第一次使用

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **运行基础示例**
   ```bash
   python 01_basic_text_to_image.py
   ```
   首次运行会自动下载模型（约 4-5GB），请耐心等待。

3. **查看生成结果**
   生成的图片保存在 `outputs/basic_text_to_image/` 目录下。

### 基本工作流程

#### 文本生成图片

```python
from 01_basic_text_to_image import generate_image_from_text

# 简单使用
generate_image_from_text(
    prompt="a beautiful sunset over the ocean",
    output_name="my_image"
)
```

#### 图片生成图片

```python
from 02_basic_image_to_image import generate_image_from_image

# 需要先准备输入图片
generate_image_from_image(
    image_path="examples/input.jpg",
    prompt="oil painting style",
    strength=0.7,
    output_name="converted_image"
)
```

## ✍️ 提示词编写指南

### 提示词结构

好的提示词通常包含以下部分：

```
[主体] + [细节描述] + [风格] + [质量修饰词]
```

**示例：**
```
"a majestic lion standing on a rock at sunset, golden hour lighting, 
photorealistic, detailed fur, 4k, high quality"
```

### 提示词技巧

#### 1. 具体化描述

- ✅ **好**：`"a red sports car on a mountain road, sunset, dramatic lighting"`
- ❌ **差**：`"a car"`

#### 2. 添加风格关键词

**艺术风格：**
- `"oil painting"`
- `"watercolor"`
- `"anime style"`
- `"cyberpunk"`
- `"impressionist"`
- `"surrealist"`

**摄影风格：**
- `"photorealistic"`
- `"cinematic"`
- `"studio lighting"`
- `"bokeh"`
- `"long exposure"`

#### 3. 质量修饰词

- `"high quality"`
- `"detailed"`
- `"4k"`
- `"8k"`
- `"professional"`
- `"sharp focus"`

#### 4. 使用负面提示词

排除不想要的内容：

```python
negative_prompt = "blurry, low quality, distorted, ugly, deformed, 
                   bad anatomy, bad proportions, watermark, text"
```

#### 5. 权重调整（高级）

使用括号调整关键词权重：

- `(keyword:1.2)` - 增加权重 20%
- `(keyword:0.8)` - 减少权重 20%
- `[keyword]` - 减少权重

**示例：**
```
"a cat, (cute:1.3), cartoon style, (colorful:1.2)"
```

### 常用提示词模板

#### 风景照
```
"[场景描述], [时间/天气], [风格], [质量修饰词]"

示例：
"a mountain landscape with a lake, sunset, golden hour, 
photorealistic, detailed, 4k"
```

#### 人物照
```
"[人物描述], [动作/姿势], [背景], [风格], [质量修饰词]"

示例：
"a portrait of a young woman, smiling, natural lighting, 
photorealistic, detailed, high quality"
```

#### 产品照
```
"[产品描述], [背景], [灯光], [风格], [质量修饰词]"

示例：
"a modern minimalist chair, white background, studio lighting, 
product photography, high quality, professional"
```

## ⚙️ 参数调优指南

### 推理步数 (num_inference_steps)

控制生成质量和速度的平衡：

| 步数 | 质量 | 速度 | 适用场景 |
|------|------|------|----------|
| 20-30 | 较低 | 快 | 快速预览、测试 |
| 50 | 良好 | 中等 | **推荐日常使用** |
| 80-100 | 最高 | 慢 | 最终作品、高质量需求 |

**建议：**
- 日常使用：50 步
- 快速测试：30 步
- 高质量输出：80 步

### 引导强度 (guidance_scale)

控制模型遵循提示词的程度：

| 强度 | 效果 | 适用场景 |
|------|------|----------|
| 1-5 | 创意性强，可能偏离提示词 | 探索性创作 |
| 7-9 | 平衡（推荐） | **日常使用** |
| 10-20 | 严格遵循提示词 | 精确控制需求 |

**建议：**
- 默认：7.5
- 需要更多创意：5-7
- 需要精确控制：9-12

### 修改强度 (strength) - 仅图片生成图片

控制对原图的修改程度：

| 强度 | 效果 | 适用场景 |
|------|------|----------|
| 0.3-0.5 | 轻微修改，保持原图结构 | 风格微调、色彩调整 |
| 0.6-0.8 | 中等修改（推荐） | **风格转换、添加元素** |
| 0.9-1.0 | 大幅修改，接近重新生成 | 完全风格转换 |

**建议：**
- 风格转换：0.6-0.8
- 轻微调整：0.4-0.5
- 大幅改变：0.8-0.9

### 图片尺寸

| 尺寸 | 内存占用 | 生成时间 | 适用场景 |
|------|----------|----------|----------|
| 512x512 | 低 | 快 | **推荐，日常使用** |
| 768x768 | 中 | 中 | 高分辨率需求 |
| 1024x1024 | 高 | 慢 | 专业作品 |

**注意：** 尺寸必须是 8 的倍数。

### 随机种子 (seed)

用于复现相同的结果：

```python
# 生成图片
generator.generate(prompt="...", seed=42, output_name="image1")

# 使用相同种子会得到相同结果
generator.generate(prompt="...", seed=42, output_name="image2")
```

**技巧：**
- 找到满意的结果后，记录使用的 seed
- 可以微调提示词，保持 seed 不变，观察变化

## 🎨 实际应用场景

### 场景1：概念设计

**目标：** 快速生成设计概念图

```python
generator = AdvancedTextToImage()

generator.generate(
    prompt="a futuristic electric car, sleek design, modern, 
            white background, product photography, high quality",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5,
    output_name="car_concept"
)
```

### 场景2：艺术创作

**目标：** 生成艺术作品

```python
generator.generate(
    prompt="a serene Japanese garden, cherry blossoms, 
            traditional architecture, watercolor style, artistic",
    negative_prompt="ugly, distorted, low quality",
    num_inference_steps=60,
    guidance_scale=8.0,
    output_name="artwork"
)
```

### 场景3：风格转换

**目标：** 将照片转换为艺术风格

```python
img_generator = AdvancedImageToImage()

img_generator.generate(
    image_path="examples/photo.jpg",
    prompt="oil painting style, Van Gogh, artistic brushstrokes",
    negative_prompt="blurry, low quality",
    strength=0.7,
    num_inference_steps=50,
    output_name="oil_painting"
)
```

### 场景4：添加元素

**目标：** 在现有图片中添加新元素

```python
img_generator.generate(
    image_path="examples/landscape.jpg",
    prompt="add a beautiful rainbow in the sky, photorealistic",
    negative_prompt="distorted, unrealistic",
    strength=0.5,  # 较低强度，保持原图结构
    output_name="landscape_rainbow"
)
```

### 场景5：季节变换

**目标：** 改变图片的季节

```python
img_generator.generate(
    image_path="examples/summer.jpg",
    prompt="winter scene, snow covering everything, cold atmosphere, peaceful",
    strength=0.7,
    output_name="winter_scene"
)
```

## 🚀 性能优化

### 1. 使用 GPU

确保安装了 CUDA 版本的 PyTorch：

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 内存优化

- **启用注意力切片**（代码已自动处理）
- **使用 float16**（GPU 上自动使用）
- **减小图片尺寸**（512x512 而不是 768x768）
- **减少推理步数**（50 而不是 100）

### 3. 批量生成

如果需要生成多张图片，考虑：

```python
prompts = [
    "a cat",
    "a dog",
    "a bird"
]

generator = AdvancedTextToImage()
generator.load_model()  # 只加载一次模型

for i, prompt in enumerate(prompts):
    generator.generate(prompt=prompt, output_name=f"image_{i}")
```

### 4. 模型缓存

模型会自动缓存到 `~/.cache/huggingface/`，后续运行无需重新下载。

## 🔧 故障排除

### 问题1：内存不足 (Out of Memory)

**解决方案：**
1. 减小图片尺寸（512x512）
2. 减少推理步数（30-50）
3. 使用 CPU 模式（虽然慢但内存占用小）
4. 关闭其他占用内存的程序

### 问题2：生成速度慢

**解决方案：**
1. 使用 GPU（如果有）
2. 减少推理步数
3. 使用较小的图片尺寸
4. 确保使用 float16（GPU 上自动）

### 问题3：生成的图片质量不好

**解决方案：**
1. 增加推理步数（50-80）
2. 使用更详细的提示词
3. 添加负面提示词
4. 尝试不同的随机种子
5. 调整引导强度（7-9）

### 问题4：模型下载失败

**解决方案：**
1. 检查网络连接
2. 使用 VPN 或代理
3. 手动下载模型到本地
4. 使用镜像站点

### 问题5：CUDA 错误

**解决方案：**
1. 检查 CUDA 版本是否匹配
2. 重新安装 PyTorch（CUDA 版本）
3. 使用 CPU 模式作为备选

## 💡 进阶技巧

### 1. 迭代优化

生成图片后，根据结果调整提示词：

```python
# 第一版
result1 = generator.generate(
    prompt="a cat",
    output_name="cat_v1"
)

# 根据结果优化
result2 = generator.generate(
    prompt="a cute orange tabby cat, sitting, detailed fur, 
            natural lighting, high quality",
    output_name="cat_v2"
)
```

### 2. 组合使用

先文本生成，再图片转换：

```python
# 步骤1: 生成基础图片
image, path = generator.text_to_image(
    prompt="a landscape",
    output_name="base"
)

# 步骤2: 风格转换
converted_image, converted_path = generator.image_to_image(
    image_path=path,
    prompt="oil painting style",
    strength=0.7,
    output_name="converted"
)
```

### 3. 参数实验

系统化测试不同参数：

```python
strengths = [0.5, 0.7, 0.9]

for strength in strengths:
    generator.image_to_image(
        image_path="input.jpg",
        prompt="anime style",
        strength=strength,
        output_name=f"anime_strength_{strength}"
    )
```

## 📖 学习资源

- [Stable Diffusion 官方文档](https://huggingface.co/docs/diffusers)
- [提示词工程指南](https://github.com/Microsoft/prompt-engineering)
- [Diffusers 库文档](https://huggingface.co/docs/diffusers/index)

---

**提示：** 实践是最好的学习方式。多尝试不同的提示词和参数组合，你会逐渐掌握技巧！

