"""
视频识别模块
"""

from .01_basic_video_recognition import classify_video, detect_objects_in_video
from .02_advanced_video_recognition import AdvancedVideoRecognition

__all__ = ['classify_video', 'detect_objects_in_video', 'AdvancedVideoRecognition']

