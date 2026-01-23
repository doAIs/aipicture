"""
Face Recognition Module - Package init file
"""
from . import (
    basic_face_recognition as basic,
    advanced_face_recognition as advanced
)

# Import functions/classes for easier access
try:
    from .basic_face_recognition import (
        detect_faces,
        encode_face,
        encode_and_save_face,
        recognize_faces,
        get_known_encodings,
        list_known_faces,
        delete_known_face,
        draw_faces_on_image
    )
except ImportError:
    pass

try:
    from .advanced_face_recognition import AdvancedFaceRecognition
except ImportError:
    pass

__all__ = [
    # Basic functions
    "detect_faces",
    "encode_face", 
    "encode_and_save_face",
    "recognize_faces",
    "get_known_encodings",
    "list_known_faces",
    "delete_known_face",
    "draw_faces_on_image",
    # Advanced class
    "AdvancedFaceRecognition"
]