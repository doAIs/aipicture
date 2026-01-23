"""
AI Multimedia Platform - FastAPI Backend
Main application entry point
"""
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from backend.core.config import settings
from backend.api.routes import router as api_router
from backend.api.websocket.training_ws import training_websocket
from backend.api.websocket.camera_ws import camera_websocket


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Mount static files for outputs
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include API routes
app.include_router(api_router, prefix="/api")


# WebSocket endpoints
@app.websocket("/ws/training/{task_id}")
async def websocket_training(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for training progress updates"""
    await training_websocket(websocket, task_id)


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """WebSocket endpoint for camera streaming"""
    await camera_websocket(websocket)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_prefix": "/api"
    }


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "An internal error occurred"
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("=" * 60)
    print(f"  {settings.API_TITLE} v{settings.API_VERSION}")
    print("=" * 60)
    print(f"  API Documentation: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"  ReDoc: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    print("=" * 60)
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  GPU: Not available (using CPU)")
    print("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("Shutting down AI Multimedia Platform API...")
    
    # Clean up resources
    from backend.core.dependencies import get_model_manager
    model_manager = get_model_manager()
    model_manager.unload_all()
    
    print("Cleanup complete. Goodbye!")


def main():
    """Run the application"""
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        workers=1  # Single worker for GPU memory management
    )


if __name__ == "__main__":
    main()
