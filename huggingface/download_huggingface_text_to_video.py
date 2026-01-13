"""
下载文本生成视频模型
使用优化的下载工具，支持进度显示、断点续传、错误重试等功能
"""

import os
import sys

from utils.download_utils import download_model

# 添加当前目录到路径，以便导入工具模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 配置参数
REPO_ID = "ali-vilab/text-to-video-ms-1.7b"
LOCAL_DIR = os.getenv("LOCAL_VIDEO_MODEL_PATH", "F:\\modules\\text-to-video\\damo")

# 可选：只下载必要的文件（取消注释以启用）
# ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer/*", "scheduler/*"]
# IGNORE_PATTERNS = ["*.ckpt", "*.pt", "*fp16*", "*training*"]

if __name__ == "__main__":
    try:
        # 下载模型（使用默认配置，下载所有文件）
        download_model(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            # allow_patterns=ALLOW_PATTERNS,  # 取消注释以只下载指定文件
            # ignore_patterns=IGNORE_PATTERNS,  # 取消注释以忽略指定文件
            max_workers=16,  # 多线程加速
            resume_download=True,  # 支持断点续传
            check_before_download=True,  # 下载前检查本地是否已存在
            retry_times=3  # 失败重试次数
        )
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)