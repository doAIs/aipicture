"""
Core module - Configuration and dependencies
"""
from .config import settings
from .dependencies import get_device, get_model_manager

__all__ = ["settings", "get_device", "get_model_manager"]
