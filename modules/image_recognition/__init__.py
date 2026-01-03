"""
图片识别模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_image_recognition', __package__)
_advanced_module = importlib.import_module('.02_advanced_image_recognition', __package__)

classify_image = _basic_module.classify_image
detect_objects = _basic_module.detect_objects
AdvancedImageRecognition = _advanced_module.AdvancedImageRecognition

__all__ = ['classify_image', 'detect_objects', 'AdvancedImageRecognition']

