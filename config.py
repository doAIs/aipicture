"""
配置文件 - 统一管理项目参数
"""
import os

# 模型配置
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"  # 默认使用的模型
# 其他可选模型：
# "stabilityai/stable-diffusion-2-1"
# "CompVis/stable-diffusion-v1-4"

# 生成参数默认值
DEFAULT_STEPS = 50  # 推理步数（越多质量越好但越慢）
DEFAULT_GUIDANCE_SCALE = 7.5  # 引导强度（越高越遵循提示词）
DEFAULT_HEIGHT = 512  # 图片高度
DEFAULT_WIDTH = 512  # 图片宽度
DEFAULT_SEED = None  # 随机种子（None表示随机）

# 输出配置
OUTPUT_DIR = "outputs"  # 输出目录
EXAMPLES_DIR = "examples"  # 示例图片目录

# 设备配置
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "true") == "true" else "cpu"

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

