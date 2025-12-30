"""
图片识别模块
"""

from .01_basic_image_recognition import classify_image, detect_objects
from .02_advanced_image_recognition import AdvancedImageRecognition

__all__ = ['classify_image', 'detect_objects', 'AdvancedImageRecognition']

