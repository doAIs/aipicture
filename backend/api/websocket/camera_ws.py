"""
Camera WebSocket - Real-time camera streaming and face recognition
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Optional
import asyncio
import json
import base64
import cv2
import numpy as np


class CameraConnectionManager:
    """Manages WebSocket connections for camera streaming"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        self.face_detection_enabled = False
        self.face_recognition_enabled = False
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        if not self.active_connections:
            self.stop_camera()
    
    def start_camera(self, camera_index: int = 0):
        """Start the camera capture"""
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_streaming = True
    
    def stop_camera(self):
        """Stop the camera capture"""
        self.is_streaming = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    async def broadcast_frame(self, frame_data: dict):
        """Send frame to all connected clients"""
        message = json.dumps(frame_data)
        dead_connections = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.add(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.active_connections.discard(conn)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera"""
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 encoded JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')


# Global camera manager
camera_manager = CameraConnectionManager()


async def camera_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for camera streaming.
    
    Usage:
        ws://localhost:8000/ws/camera
    
    Client commands:
        {"type": "start", "camera_index": 0}
        {"type": "stop"}
        {"type": "enable_face_detection"}
        {"type": "disable_face_detection"}
        {"type": "enable_face_recognition"}
        {"type": "disable_face_recognition"}
    
    Messages sent:
        {
            "type": "frame",
            "data": "base64_encoded_jpeg",
            "faces": [{"x": 100, "y": 100, "width": 50, "height": 50, "name": "John"}],
            "timestamp": 1234567890
        }
    """
    await camera_manager.connect(websocket)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to camera stream"
        })
        
        while True:
            try:
                # Wait for client commands
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.05  # 50ms timeout for smooth streaming
                )
                
                message = json.loads(data)
                await handle_camera_command(websocket, message)
                
            except asyncio.TimeoutError:
                # Stream frame if camera is active
                if camera_manager.is_streaming:
                    await stream_frame(websocket)
                    
    except WebSocketDisconnect:
        camera_manager.disconnect(websocket)
    except Exception as e:
        camera_manager.disconnect(websocket)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


async def handle_camera_command(websocket: WebSocket, message: dict):
    """Handle camera control commands from client"""
    msg_type = message.get("type")
    
    if msg_type == "start":
        camera_index = message.get("camera_index", 0)
        camera_manager.start_camera(camera_index)
        await websocket.send_json({
            "type": "started",
            "message": f"Camera {camera_index} started"
        })
        
    elif msg_type == "stop":
        camera_manager.stop_camera()
        await websocket.send_json({
            "type": "stopped",
            "message": "Camera stopped"
        })
        
    elif msg_type == "enable_face_detection":
        camera_manager.face_detection_enabled = True
        await websocket.send_json({
            "type": "face_detection_enabled",
            "message": "Face detection enabled"
        })
        
    elif msg_type == "disable_face_detection":
        camera_manager.face_detection_enabled = False
        await websocket.send_json({
            "type": "face_detection_disabled",
            "message": "Face detection disabled"
        })
        
    elif msg_type == "enable_face_recognition":
        camera_manager.face_recognition_enabled = True
        camera_manager.face_detection_enabled = True  # Also enable detection
        await websocket.send_json({
            "type": "face_recognition_enabled",
            "message": "Face recognition enabled"
        })
        
    elif msg_type == "disable_face_recognition":
        camera_manager.face_recognition_enabled = False
        await websocket.send_json({
            "type": "face_recognition_disabled",
            "message": "Face recognition disabled"
        })
        
    elif msg_type == "ping":
        await websocket.send_json({"type": "pong"})


async def stream_frame(websocket: WebSocket):
    """Capture and stream a single frame"""
    import time
    
    frame = camera_manager.capture_frame()
    if frame is None:
        return
    
    faces = []
    
    # Face detection/recognition if enabled
    if camera_manager.face_detection_enabled:
        try:
            faces = detect_faces_in_frame(frame, camera_manager.face_recognition_enabled)
            
            # Draw rectangles on faces
            for face in faces:
                x, y, w, h = face["x"], face["y"], face["width"], face["height"]
                color = (0, 255, 0) if face.get("name") else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                if face.get("name"):
                    cv2.putText(frame, face["name"], (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception:
            pass
    
    # Convert frame to base64
    frame_b64 = camera_manager.frame_to_base64(frame)
    
    # Send frame
    await websocket.send_json({
        "type": "frame",
        "data": frame_b64,
        "faces": faces,
        "timestamp": int(time.time() * 1000)
    })


def detect_faces_in_frame(frame: np.ndarray, recognize: bool = False) -> list:
    """Detect faces in a frame using OpenCV or face_recognition library"""
    faces = []
    
    try:
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use face_recognition library if available
        try:
            import face_recognition as fr
            
            # Find face locations
            face_locations = fr.face_locations(rgb_frame, model="hog")
            
            for (top, right, bottom, left) in face_locations:
                face_data = {
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top
                }
                
                # Recognize if enabled
                if recognize:
                    try:
                        from modules.face_recognition import get_known_encodings
                        known_encodings, known_names = get_known_encodings()
                        
                        if known_encodings:
                            face_encoding = fr.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
                            matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                            
                            if True in matches:
                                match_index = matches.index(True)
                                face_data["name"] = known_names[match_index]
                    except Exception:
                        pass
                
                faces.append(face_data)
                
        except ImportError:
            # Fallback to OpenCV Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            detected = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in detected:
                faces.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
                
    except Exception:
        pass
    
    return faces
