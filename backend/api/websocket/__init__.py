"""
WebSocket module - Real-time communication endpoints
"""
from .training_ws import training_websocket
from .camera_ws import camera_websocket

__all__ = ["training_websocket", "camera_websocket"]
