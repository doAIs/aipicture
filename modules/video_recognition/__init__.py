"""
视频识别模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_video_recognition', __package__)
_advanced_module = importlib.import_module('.02_advanced_video_recognition', __package__)

classify_video = _basic_module.classify_video
detect_objects_in_video = _basic_module.detect_objects_in_video
AdvancedVideoRecognition = _advanced_module.AdvancedVideoRecognition

__all__ = ['classify_video', 'detect_objects_in_video', 'AdvancedVideoRecognition']

