"""
视频生成视频模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_video_to_video', __package__)
_advanced_module = importlib.import_module('.02_advanced_video_to_video', __package__)

generate_video_from_video = _basic_module.generate_video_from_video
AdvancedVideoToVideo = _advanced_module.AdvancedVideoToVideo

__all__ = ['generate_video_from_video', 'AdvancedVideoToVideo']

