# 示例图片目录

此目录用于存放输入图片，用于图片生成图片（Image-to-Image）功能。

## 📁 目录说明

将你想要转换的图片放在此目录下，然后在代码中引用。

## 📝 使用示例

### 准备输入图片

1. 将你的图片文件（支持 JPG、PNG 等格式）复制到此目录
2. 建议使用清晰、高质量的图片以获得更好的效果
3. 推荐尺寸：512x512 或 768x768（代码会自动调整）

### 在代码中使用

```python
from 02_basic_image_to_image import generate_image_from_image

# 使用此目录下的图片
generate_image_from_image(
    image_path="examples/your_image.jpg",  # 替换为你的图片文件名
    prompt="oil painting style, artistic, detailed",
    strength=0.7,
    output_name="converted_image"
)
```

## 🎨 推荐的图片类型

- **风景照**：适合风格转换（油画、水彩、动漫等）
- **人物照片**：适合风格转换和艺术化处理
- **建筑照片**：适合风格转换和场景变换
- **静物照片**：适合添加元素和风格转换

## ⚠️ 注意事项

1. **图片格式**：支持 JPG、PNG、JPEG 等常见格式
2. **图片大小**：建议不超过 2048x2048，代码会自动调整
3. **图片质量**：使用高质量、清晰的图片效果更好
4. **版权**：请确保使用的图片有合法使用权

## 📌 示例场景

### 场景1：风格转换
- 输入：普通照片
- 提示词：`"oil painting style, artistic, detailed brushstrokes"`
- 强度：0.5-0.7

### 场景2：季节变换
- 输入：夏季风景照
- 提示词：`"winter scene, snow, cold atmosphere, peaceful"`
- 强度：0.6-0.8

### 场景3：添加元素
- 输入：风景照
- 提示词：`"add a beautiful rainbow in the sky, photorealistic"`
- 强度：0.5-0.6

### 场景4：动漫化
- 输入：真实照片
- 提示词：`"anime style, vibrant colors, detailed, studio ghibli"`
- 强度：0.7-0.9

---

**提示**：如果没有输入图片，可以先运行 `01_basic_text_to_image.py` 生成一些图片作为输入。

