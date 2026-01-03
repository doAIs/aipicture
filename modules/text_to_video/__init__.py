"""
文本生成视频模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_text_to_video', __package__)
_advanced_module = importlib.import_module('.02_advanced_text_to_video', __package__)

generate_video_from_text = _basic_module.generate_video_from_text
AdvancedTextToVideo = _advanced_module.AdvancedTextToVideo

__all__ = ['generate_video_from_text', 'AdvancedTextToVideo']

