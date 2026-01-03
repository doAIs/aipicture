"""
文本生成图片模块
"""

import importlib

_basic_module = importlib.import_module('.01_basic_text_to_image', __package__)
_advanced_module = importlib.import_module('.02_advanced_text_to_image', __package__)

generate_image_from_text = _basic_module.generate_image_from_text
AdvancedTextToImage = _advanced_module.AdvancedTextToImage

__all__ = ['generate_image_from_text', 'AdvancedTextToImage']

