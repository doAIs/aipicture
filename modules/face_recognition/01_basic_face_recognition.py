"""
Face Recognition Module - Face detection and recognition functionality
"""
import face_recognition
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


def detect_faces(image_path: str, model: str = "hog") -> List[Dict]:
    """
    Detect faces in an image using face_recognition library
    
    Args:
        image_path: Path to the image file
        model: Detection model ('hog' or 'cnn')
    
    Returns:
        List of dictionaries containing face locations
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image, model=model)
    
    # Convert to our format
    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append({
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top
        })
    
    return faces


def encode_face(image_path: str) -> List[np.ndarray]:
    """
    Encode faces in an image to face encodings
    
    Args:
        image_path: Path to the image file
    
    Returns:
        List of face encodings
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Encode faces
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings


def encode_and_save_face(image_path: str, name: str, face_db_path: str = "data/faces") -> Dict:
    """
    Encode a face and save to the face database
    
    Args:
        image_path: Path to the image file
        name: Name to associate with the face
        face_db_path: Path to the face database directory
    
    Returns:
        Dictionary with result information
    """
    # Create face database directory if it doesn't exist
    os.makedirs(face_db_path, exist_ok=True)
    
    # Encode face
    encodings = encode_face(image_path)
    
    if not encodings:
        return {"success": False, "message": "No faces found in the image"}
    
    # Use the first face if multiple are detected
    encoding = encodings[0]
    
    # Save encoding
    face_db_file = os.path.join(face_db_path, f"{name}.pkl")
    with open(face_db_file, 'wb') as f:
        pickle.dump(encoding, f)
    
    return {
        "success": True,
        "message": f"Face encoded and saved as '{name}'",
        "encoding_shape": encoding.shape,
        "saved_path": face_db_file
    }


def recognize_faces(image_path: str, tolerance: float = 0.6, face_db_path: str = "data/faces") -> List[Dict]:
    """
    Recognize faces in an image by comparing with known faces
    
    Args:
        image_path: Path to the image file
        tolerance: Recognition tolerance (lower = stricter)
        face_db_path: Path to the face database directory
    
    Returns:
        List of recognized faces with names and locations
    """
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations and encodings
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Load known faces
    known_encodings, known_names = get_known_encodings(face_db_path)
    
    if not known_encodings:
        return [{"x": loc[3], "y": loc[0], "width": loc[1]-loc[3], "height": loc[2]-loc[0], 
                "name": "unknown", "confidence": 0.0} 
                for loc in face_locations]
    
    # Compare faces
    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        name = "unknown"
        confidence = 0.0
        
        if True in matches:
            # Get the closest match
            match_index = matches.index(True)
            min_distance = distances[match_index]
            name = known_names[match_index]
            confidence = max(0.0, 1.0 - min_distance)  # Convert distance to confidence
        
        results.append({
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
            "name": name,
            "confidence": confidence
        })
    
    return results


def get_known_encodings(face_db_path: str = "data/faces") -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all known face encodings from the database
    
    Args:
        face_db_path: Path to the face database directory
    
    Returns:
        Tuple of (encodings_list, names_list)
    """
    known_encodings = []
    known_names = []
    
    if not os.path.exists(face_db_path):
        return known_encodings, known_names
    
    for filename in os.listdir(face_db_path):
        if filename.endswith('.pkl'):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(face_db_path, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    encoding = pickle.load(f)
                    known_encodings.append(encoding)
                    known_names.append(name)
            except Exception as e:
                print(f"Error loading face encoding {filename}: {e}")
    
    return known_encodings, known_names


def list_known_faces(face_db_path: str = "data/faces") -> List[str]:
    """
    List all known faces in the database
    
    Args:
        face_db_path: Path to the face database directory
    
    Returns:
        List of face names
    """
    if not os.path.exists(face_db_path):
        return []
    
    faces = []
    for filename in os.listdir(face_db_path):
        if filename.endswith('.pkl'):
            name = os.path.splitext(filename)[0]
            faces.append(name)
    
    return faces


def delete_known_face(name: str, face_db_path: str = "data/faces") -> bool:
    """
    Delete a known face from the database
    
    Args:
        name: Name of the face to delete
        face_db_path: Path to the face database directory
    
    Returns:
        True if deletion was successful, False otherwise
    """
    face_file = os.path.join(face_db_path, f"{name}.pkl")
    
    if os.path.exists(face_file):
        try:
            os.remove(face_file)
            return True
        except Exception as e:
            print(f"Error deleting face {name}: {e}")
            return False
    else:
        return False


def draw_faces_on_image(image_path: str, output_path: str = None, model: str = "hog") -> str:
    """
    Draw rectangles around detected faces on an image
    
    Args:
        image_path: Path to the input image
        output_path: Path for the output image (optional)
        model: Detection model ('hog' or 'cnn')
    
    Returns:
        Path to the output image
    """
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_faces{ext}"
    
    # Load image
    image = face_recognition.load_image_file(image_path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces = detect_faces(image_path, model)
    
    # Draw rectangles
    for face in faces:
        x, y, w, h = face["x"], face["y"], face["width"], face["height"]
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save image
    cv2.imwrite(output_path, image_bgr)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Face Recognition Module")
    print("=" * 30)
    
    # Example: Detect faces in an image
    # image_path = "path/to/your/image.jpg"  # Replace with actual image path
    # faces = detect_faces(image_path)
    # print(f"Detected {len(faces)} faces")
    # for i, face in enumerate(faces):
    #     print(f"Face {i+1}: x={face['x']}, y={face['y']}, w={face['width']}, h={face['height']}")
    
    print("Module ready for face recognition tasks")