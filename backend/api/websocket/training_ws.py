"""
Training WebSocket - Real-time training progress updates
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json

# Active WebSocket connections for training updates
active_connections: Dict[str, Set[WebSocket]] = {}


class TrainingConnectionManager:
    """Manages WebSocket connections for training progress updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """Accept a new WebSocket connection for a training task"""
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        self.active_connections[task_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        """Remove a WebSocket connection"""
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
    
    async def send_progress(self, task_id: str, data: dict):
        """Send progress update to all connections for a task"""
        if task_id in self.active_connections:
            message = json.dumps(data)
            dead_connections = set()
            
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.active_connections[task_id].discard(conn)
    
    async def broadcast_all(self, data: dict):
        """Broadcast message to all active connections"""
        message = json.dumps(data)
        for task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    pass


# Global connection manager
training_manager = TrainingConnectionManager()


async def training_websocket(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for training progress updates.
    Clients connect to receive real-time training metrics.
    
    Usage:
        ws://localhost:8000/ws/training/{task_id}
    
    Messages sent:
        {
            "type": "progress",
            "task_id": "...",
            "progress": 45.5,
            "epoch": 2,
            "step": 150,
            "loss": 0.234,
            "metrics": {...}
        }
    """
    await training_manager.connect(websocket, task_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "task_id": task_id,
            "message": "Connected to training progress stream"
        })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for messages from client (heartbeat, commands, etc.)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout for heartbeat
                )
                
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    # Client can subscribe to additional task updates
                    new_task_id = message.get("task_id")
                    if new_task_id:
                        if new_task_id not in training_manager.active_connections:
                            training_manager.active_connections[new_task_id] = set()
                        training_manager.active_connections[new_task_id].add(websocket)
                        await websocket.send_json({
                            "type": "subscribed",
                            "task_id": new_task_id
                        })
                        
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        training_manager.disconnect(websocket, task_id)
    except Exception as e:
        training_manager.disconnect(websocket, task_id)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


async def send_training_update(task_id: str, progress: float, epoch: int = 0, 
                               step: int = 0, loss: float = None, metrics: dict = None):
    """
    Helper function to send training updates to connected clients.
    Call this from your training loop.
    """
    await training_manager.send_progress(task_id, {
        "type": "progress",
        "task_id": task_id,
        "progress": progress,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "metrics": metrics or {}
    })


async def send_training_complete(task_id: str, result: dict):
    """Send training completion message"""
    await training_manager.send_progress(task_id, {
        "type": "complete",
        "task_id": task_id,
        "result": result
    })


async def send_training_error(task_id: str, error: str):
    """Send training error message"""
    await training_manager.send_progress(task_id, {
        "type": "error",
        "task_id": task_id,
        "error": error
    })
