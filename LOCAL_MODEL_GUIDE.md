# 本地模型加载指南

本指南说明如何加载本地下载的 Stable Diffusion 模型文件。

## 支持的模型格式

- `.safetensors` 文件（推荐）
- `.ckpt` 文件
- 模型目录（包含多个文件的完整模型目录）

## 使用方法

### 方法1: 在代码中直接指定本地模型路径

```python
from modules.text_to_image import generate_image_from_text

# 指定本地模型文件路径
local_model_path = "v1-5-pruned.safetensors"  # 相对路径
# 或使用绝对路径
# local_model_path = "E:/models/v1-5-pruned.safetensors"

prompt = "a beautiful sunset over the ocean"
generate_image_from_text(
    prompt=prompt,
    output_name="output_image",
    model_path=local_model_path  # 传入本地模型路径
)
```

### 方法2: 使用 utils 中的函数直接加载

```python
from diffusers import StableDiffusionPipeline
from utils.modules_utils import load_model_from_local_file
import torch

# 加载本地模型
pipe = load_model_from_local_file(
    StableDiffusionPipeline,
    "v1-5-pruned.safetensors",  # 您的模型文件路径
    torch_dtype=torch.float16,  # 或 torch.float32
    safety_checker=None,
    requires_safety_checker=False
)

# 使用模型生成图片
image = pipe("a beautiful sunset").images[0]
```

### 方法3: 修改配置文件

您也可以修改 `config.py` 中的 `DEFAULT_IMAGE_MODEL` 为本地路径：

```python
# 在 modules_config.py 中
DEFAULT_IMAGE_MODEL = "E:/models/v1-5-pruned.safetensors"
```

## 模型文件位置

将您的模型文件放在以下任一位置：

1. **项目根目录**：直接放在项目文件夹中
   ```
   aipicture/
   ├── v1-5-pruned.safetensors
   ├── modules/
   └── ...
   ```

2. **专门的模型目录**：创建一个 models 文件夹
   ```
   aipicture/
   ├── models/
   │   └── v1-5-pruned.safetensors
   ├── modules/
   └── ...
   ```

3. **任意位置**：使用绝对路径指定

## 注意事项

1. **文件路径**：确保路径正确，可以使用相对路径或绝对路径
2. **文件格式**：确保文件扩展名是 `.safetensors` 或 `.ckpt`
3. **diffusers 版本**：需要 diffusers >= 0.21.0 才能使用 `from_single_file` 方法
4. **内存要求**：本地模型加载可能需要较多内存，建议使用 GPU

## 常见问题

### Q: 提示 "不支持的文件格式"
A: 确保文件扩展名是 `.safetensors` 或 `.ckpt`，且文件完整未损坏。

### Q: 提示 "模型文件不存在"
A: 检查文件路径是否正确，可以使用绝对路径避免路径问题。

### Q: 加载速度很慢
A: 首次加载需要将模型加载到内存，这是正常的。后续使用会更快。

## 示例代码

完整示例请参考 `example_load_local_model.py` 文件。



# 本地模型加载指南

本指南说明如何加载本地下载的 Stable Diffusion 模型文件。

## 支持的模型格式

- `.safetensors` 文件（推荐）
- `.ckpt` 文件
- 模型目录（包含多个文件的完整模型目录）

## 使用方法

### 方法1: 在代码中直接指定本地模型路径

```python
from modules.text_to_image import generate_image_from_text

# 指定本地模型文件路径
local_model_path = "v1-5-pruned.safetensors"  # 相对路径
# 或使用绝对路径
# local_model_path = "E:/models/v1-5-pruned.safetensors"

prompt = "a beautiful sunset over the ocean"
generate_image_from_text(
    prompt=prompt,
    output_name="output_image",
    model_path=local_model_path  # 传入本地模型路径
)
```

### 方法2: 使用 utils 中的函数直接加载

```python
from diffusers import StableDiffusionPipeline
from utils.modules_utils import load_model_from_local_file
import torch

# 加载本地模型
pipe = load_model_from_local_file(
    StableDiffusionPipeline,
    "v1-5-pruned.safetensors",  # 您的模型文件路径
    torch_dtype=torch.float16,  # 或 torch.float32
    safety_checker=None,
    requires_safety_checker=False
)

# 使用模型生成图片
image = pipe("a beautiful sunset").images[0]
```

### 方法3: 修改配置文件

您也可以修改 `config.py` 中的 `DEFAULT_IMAGE_MODEL` 为本地路径：

```python
# 在 modules_config.py 中
DEFAULT_IMAGE_MODEL = "E:/models/v1-5-pruned.safetensors"
```

## 模型文件位置

将您的模型文件放在以下任一位置：

1. **项目根目录**：直接放在项目文件夹中
   ```
   aipicture/
   ├── v1-5-pruned.safetensors
   ├── modules/
   └── ...
   ```

2. **专门的模型目录**：创建一个 models 文件夹
   ```
   aipicture/
   ├── models/
   │   └── v1-5-pruned.safetensors
   ├── modules/
   └── ...
   ```

3. **任意位置**：使用绝对路径指定

## 注意事项

1. **文件路径**：确保路径正确，可以使用相对路径或绝对路径
2. **文件格式**：确保文件扩展名是 `.safetensors` 或 `.ckpt`
3. **diffusers 版本**：需要 diffusers >= 0.21.0 才能使用 `from_single_file` 方法
4. **内存要求**：本地模型加载可能需要较多内存，建议使用 GPU

## 常见问题

### Q: 提示 "不支持的文件格式"
A: 确保文件扩展名是 `.safetensors` 或 `.ckpt`，且文件完整未损坏。

### Q: 提示 "模型文件不存在"
A: 检查文件路径是否正确，可以使用绝对路径避免路径问题。

### Q: 加载速度很慢
A: 首次加载需要将模型加载到内存，这是正常的。后续使用会更快。

## 示例代码

完整示例请参考 `example_load_local_model.py` 文件。



