# 示例文件说明

本目录用于存放示例图片和视频文件，供项目中的各种功能使用。

## 目录结构

```
examples/
├── images/          # 示例图片目录
│   ├── input.jpg    # 输入图片示例
│   └── ...
└── videos/          # 示例视频目录
    ├── input.mp4    # 输入视频示例
    └── ...
```

## 支持的图片格式

- `.jpg` / `.jpeg`
- `.png`
- `.bmp`
- `.tiff`

## 支持的视频格式

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`

## 使用建议

### 图片要求

- **尺寸**：建议 512x512 或更大（会自动调整）
- **格式**：RGB 格式（会自动转换）
- **大小**：建议小于 10MB

### 视频要求

- **分辨率**：建议 256x256 或更大
- **时长**：建议 5-30 秒（处理时间与时长成正比）
- **帧率**：建议 8-30 fps
- **大小**：建议小于 100MB

## 准备示例文件

### 方式1：手动添加

直接将你的图片或视频文件复制到对应的目录：
- 图片 → `examples/images/`
- 视频 → `examples/videos/`

### 方式2：使用项目生成

你可以先使用文本生成图片功能生成一些图片，然后用于其他功能：

```python
from modules.text_to_image import AdvancedTextToImage

generator = AdvancedTextToImage()
generator.generate(
    prompt="a beautiful landscape",
    output_name="test_image"
)
# 生成的图片在 outputs/images/ 目录下
# 可以复制到 examples/images/ 目录使用
```

## 注意事项

1. 确保文件路径正确
2. 检查文件格式是否支持
3. 注意文件大小，过大的文件可能导致内存不足
4. 视频处理需要较长时间，建议先用短视频测试
