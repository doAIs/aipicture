"""
配置文件 - 统一管理项目参数
"""
import os

# ==================== 模型配置 ====================
# 图片生成模型
DEFAULT_IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"  # 默认使用的图片生成模型
# 其他可选模型：
# "stabilityai/stable-diffusion-2-1"
# "CompVis/stable-diffusion-v1-4"

# 视频生成模型
DEFAULT_VIDEO_MODEL = "damo-vilab/text-to-video-ms-1.7b"  # 默认视频生成模型
# 其他可选模型：
# "cerspense/zeroscope-v2_576w"  # 高质量视频生成
# "stabilityai/stable-video-diffusion-img2vid"

# 图像识别模型
DEFAULT_IMAGE_RECOGNITION_MODEL = "google/vit-base-patch16-224"  # 图像分类模型
DEFAULT_OBJECT_DETECTION_MODEL = "yolov8n.pt"  # 目标检测模型（YOLOv8）

# ==================== 生成参数默认值 ====================
# 图片生成参数
DEFAULT_STEPS = 50  # 推理步数（越多质量越好但越慢）
DEFAULT_GUIDANCE_SCALE = 7.5  # 引导强度（越高越遵循提示词）
DEFAULT_HEIGHT = 512  # 图片高度
DEFAULT_WIDTH = 512  # 图片宽度
DEFAULT_SEED = None  # 随机种子（None表示随机）

# 视频生成参数
DEFAULT_VIDEO_STEPS = 50  # 视频推理步数
DEFAULT_VIDEO_FRAMES = 16  # 视频帧数
DEFAULT_VIDEO_FPS = 8  # 视频帧率
DEFAULT_VIDEO_HEIGHT = 256  # 视频高度
DEFAULT_VIDEO_WIDTH = 256  # 视频宽度

# ==================== 输出配置 ====================
OUTPUT_DIR = "outputs"  # 输出目录
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")  # 图片输出目录
OUTPUT_VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")  # 视频输出目录

EXAMPLES_DIR = "examples"  # 示例目录
EXAMPLES_IMAGES_DIR = os.path.join(EXAMPLES_DIR, "images")  # 示例图片目录
EXAMPLES_VIDEOS_DIR = os.path.join(EXAMPLES_DIR, "videos")  # 示例视频目录

# ==================== 设备配置 ====================
DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "true") == "true" else "cpu"

# ==================== 创建必要的目录 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
os.makedirs(EXAMPLES_IMAGES_DIR, exist_ok=True)
os.makedirs(EXAMPLES_VIDEOS_DIR, exist_ok=True)

