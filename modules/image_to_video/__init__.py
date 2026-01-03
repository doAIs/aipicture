"""
图片生成视频模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_image_to_video', __package__)
_advanced_module = importlib.import_module('.02_advanced_image_to_video', __package__)

generate_video_from_image = _basic_module.generate_video_from_image
AdvancedImageToVideo = _advanced_module.AdvancedImageToVideo

__all__ = ['generate_video_from_image', 'AdvancedImageToVideo']

