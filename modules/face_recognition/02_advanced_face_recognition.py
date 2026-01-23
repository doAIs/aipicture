"""
Advanced Face Recognition Module - Enhanced face recognition functionality
"""
import face_recognition
import cv2
import numpy as np
import os
import pickle
import time
from typing import List, Tuple, Dict, Optional
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.modules_utils import get_device


class AdvancedFaceRecognition:
    """
    Advanced Face Recognition Class
    Provides comprehensive face recognition functionality including:
    - Face detection
    - Face encoding and recognition
    - Face database management
    - Live camera recognition
    """
    
    def __init__(self, face_db_path: str = "data/faces", tolerance: float = 0.6):
        self.face_db_path = Path(face_db_path)
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load all known face encodings from the database"""
        self.known_encodings = []
        self.known_names = []
        
        if not self.face_db_path.exists():
            self.face_db_path.mkdir(parents=True, exist_ok=True)
            return
        
        for file_path in self.face_db_path.glob("*.pkl"):
            name = file_path.stem
            try:
                with open(file_path, 'rb') as f:
                    encoding = pickle.load(f)
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
            except Exception as e:
                print(f"Error loading face encoding {file_path.name}: {e}")
    
    def detect_faces(self, image_path: str, model: str = "hog") -> List[Dict]:
        """
        Detect faces in an image
        
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
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convert to our format
        faces = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            faces.append({
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
                "encoding": encoding.tolist() if encoding is not None else None
            })
        
        return faces
    
    def encode_face(self, image_path: str) -> List[np.ndarray]:
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
    
    def add_face_to_database(self, image_path: str, name: str) -> Dict:
        """
        Add a face to the database
        
        Args:
            image_path: Path to the image file
            name: Name to associate with the face
        
        Returns:
            Dictionary with result information
        """
        # Encode face
        encodings = self.encode_face(image_path)
        
        if not encodings:
            return {"success": False, "message": "No faces found in the image"}
        
        # Use the first face if multiple are detected
        encoding = encodings[0]
        
        # Save encoding
        face_db_file = self.face_db_path / f"{name}.pkl"
        with open(face_db_file, 'wb') as f:
            pickle.dump(encoding, f)
        
        # Reload known faces
        self._load_known_faces()
        
        return {
            "success": True,
            "message": f"Face encoded and saved as '{name}'",
            "encoding_shape": encoding.shape,
            "saved_path": str(face_db_file)
        }
    
    def recognize_faces(self, image_path: str) -> List[Dict]:
        """
        Recognize faces in an image by comparing with known faces
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of recognized faces with names and locations
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not self.known_encodings:
            # Return unknown faces if no known faces exist
            return [{"x": loc[3], "y": loc[0], "width": loc[1]-loc[3], "height": loc[2]-loc[0], 
                    "name": "unknown", "confidence": 0.0, "encoding": None} 
                    for loc in face_locations]
        
        # Compare faces
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            
            name = "unknown"
            confidence = 0.0
            
            if True in matches:
                # Get the closest match
                match_index = matches.index(True)
                min_distance = distances[match_index]
                name = self.known_names[match_index]
                confidence = max(0.0, 1.0 - min_distance)  # Convert distance to confidence
            
            results.append({
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
                "name": name,
                "confidence": confidence,
                "encoding": face_encoding.tolist()
            })
        
        return results
    
    def recognize_faces_in_real_time(self, camera_index: int = 0, show_window: bool = True) -> None:
        """
        Recognize faces in real-time using camera
        
        Args:
            camera_index: Index of the camera to use
            show_window: Whether to show the video window
        """
        # Open camera
        video_capture = cv2.VideoCapture(camera_index)
        
        if not video_capture.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time face recognition. Press 'q' to quit.")
        
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
                name = "Unknown"
                confidence = 0.0
                
                # Or instead of just using the known face with the smallest distance,
                # use the known face with the smallest distance to this face
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = max(0.0, 1.0 - face_distances[best_match_index])
                
                face_names.append((name, confidence))
            
            # Display the results
            for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            if show_window:
                # Display the resulting image
                cv2.imshow('Face Recognition', frame)
                
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
    
    def get_face_database_stats(self) -> Dict:
        """Get statistics about the face database"""
        if not self.face_db_path.exists():
            return {"count": 0, "path": str(self.face_db_path), "size_mb": 0}
        
        files = list(self.face_db_path.glob("*.pkl"))
        size_bytes = sum(f.stat().st_size for f in files)
        
        return {
            "count": len(files),
            "path": str(self.face_db_path),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "names": [f.stem for f in files]
        }
    
    def remove_face_from_database(self, name: str) -> bool:
        """Remove a face from the database"""
        face_file = self.face_db_path / f"{name}.pkl"
        
        if face_file.exists():
            try:
                face_file.unlink()
                self._load_known_faces()  # Reload known faces
                return True
            except Exception as e:
                print(f"Error removing face {name}: {e}")
                return False
        else:
            return False
    
    def update_tolerance(self, tolerance: float):
        """Update the recognition tolerance"""
        self.tolerance = tolerance


if __name__ == "__main__":
    # Example usage
    print("Advanced Face Recognition Module")
    print("=" * 40)
    
    # Initialize the recognizer
    recognizer = AdvancedFaceRecognition()
    
    # Example: Add a face to the database
    # result = recognizer.add_face_to_database("path/to/image.jpg", "john_doe")
    # print(f"Add face result: {result}")
    
    # Example: Recognize faces in an image
    # results = recognizer.recognize_faces("path/to/test_image.jpg")
    # print(f"Recognized faces: {results}")
    
    # Example: Get database stats
    stats = recognizer.get_face_database_stats()
    print(f"Face database stats: {stats}")
    
    print("Module ready for advanced face recognition tasks")