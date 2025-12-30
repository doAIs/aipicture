"""
图片生成视频模块
"""

from .01_basic_image_to_video import generate_video_from_image
from .02_advanced_image_to_video import AdvancedImageToVideo

__all__ = ['generate_video_from_image', 'AdvancedImageToVideo']

