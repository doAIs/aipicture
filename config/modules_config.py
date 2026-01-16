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

# 本地模型路径配置（优先使用本地模型）
# 可以通过环境变量 LOCAL_MODEL_PATH 或直接修改此配置来设置本地模型路径
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "F:\\modules\\sd")  # 默认本地模型路径
# 如果设置为 None 或空字符串，则不使用本地模型
# 示例：
# LOCAL_MODEL_PATH = "F:\\modules\\sd"  # Windows路径
# LOCAL_MODEL_PATH = "/home/user/models/sd"  # Linux/Mac路径
# LOCAL_MODEL_PATH = None  # 禁用本地模型，仅使用在线模型

# 视频生成模型
DEFAULT_VIDEO_MODEL = "ali-vilab/text-to-video-ms-1.7b"  # 默认视频生成模型
# 其他可选模型：
# "cerspense/zeroscope-v2_576w"  # 高质量视频生成
# "stabilityai/stable-video-diffusion-img2vid"

# 本地视频模型路径配置（优先使用本地模型）
# 可以通过环境变量 LOCAL_VIDEO_MODEL_PATH 或直接修改此配置来设置本地视频模型路径
LOCAL_VIDEO_MODEL_PATH = os.getenv("LOCAL_VIDEO_MODEL_PATH", "F:\\modules\\text-to-video\\damo")  # 默认本地视频模型路径
# 如果设置为 None 或空字符串，则不使用本地模型
# 示例：
# LOCAL_VIDEO_MODEL_PATH = "F:\\modules\\text-to-video\\damo"  # Windows路径
# LOCAL_VIDEO_MODEL_PATH = "/home/user/models/text-to-video"  # Linux/Mac路径
# LOCAL_VIDEO_MODEL_PATH = None  # 禁用本地模型，仅使用在线模型

# 图像识别模型
DEFAULT_IMAGE_RECOGNITION_MODEL = "google/vit-base-patch16-224"  # 图像分类模型
DEFAULT_OBJECT_DETECTION_MODEL = "yolov8n.pt"  # 目标检测模型（YOLOv8）

# ==================== 本地模型路径配置（按模块） ====================
# 图像识别模型本地路径
# 支持 Hugging Face 格式的模型目录（如 ViT, ResNet 等）
LOCAL_IMAGE_RECOGNITION_MODEL_PATH = os.getenv(
    "LOCAL_IMAGE_RECOGNITION_MODEL_PATH", 
    "F:\\modules\\google\\vit-base-patch16-224"
)
# 如果设置为 None 或空字符串，则不使用本地模型

# 目标检测模型本地路径（YOLO）
# 支持 .pt 文件或模型目录
LOCAL_OBJECT_DETECTION_MODEL_PATH = os.getenv(
    "LOCAL_OBJECT_DETECTION_MODEL_PATH", 
    "F:\\modules\\object-detection\\yolov8n.pt"
)
# 示例路径：
# LOCAL_OBJECT_DETECTION_MODEL_PATH = "F:\\modules\\yolov8n.pt"  # Windows
# LOCAL_OBJECT_DETECTION_MODEL_PATH = "/home/user/models/yolov8n.pt"  # Linux

# 图片生成图片模型本地路径（Img2Img）
LOCAL_IMAGE_TO_IMAGE_MODEL_PATH = os.getenv(
    "LOCAL_IMAGE_TO_IMAGE_MODEL_PATH", 
    "F:\\modules\\stable-diffusion\\sd-v1-5"
)
# 支持 .safetensors, .ckpt 文件或 Hugging Face 格式目录

# 图片生成视频模型本地路径（SVD）
LOCAL_IMAGE_TO_VIDEO_MODEL_PATH = os.getenv(
    "LOCAL_IMAGE_TO_VIDEO_MODEL_PATH", 
    "F:\\modules\\stable-video-diffusion\\svd-img2vid"
)
# 支持 .safetensors, .ckpt 文件或 Hugging Face 格式目录

# 视频生成视频模型本地路径（Video2Video）
LOCAL_VIDEO_TO_VIDEO_MODEL_PATH = os.getenv(
    "LOCAL_VIDEO_TO_VIDEO_MODEL_PATH", 
    "F:\\modules\\stable-diffusion\\sd-v1-5"
)
# 通常与 Image-to-Image 使用相同的模型（逐帧处理）

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

