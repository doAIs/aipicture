# API Reference

Complete API documentation for the AI Multimedia Platform.

## Base URL

```
http://localhost:8000/api
```

## Authentication

Most endpoints do not require authentication. For protected endpoints, include the authorization header:

```
Authorization: Bearer <token>
```

---

## Generation APIs

### Text to Image

#### Generate Image

```http
POST /text-to-image/generate
```

**Request Body:**
```json
{
  "prompt": "a beautiful sunset over the ocean",
  "negative_prompt": "blurry, low quality",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42,
  "model_id": "stable-diffusion-v1-5"
}
```

**Response:**
```json
{
  "task_id": "task_12345",
  "status": "processing",
  "image_url": null,
  "progress": 0
}
```

#### Get Generation Status

```http
GET /text-to-image/status/{task_id}
```

**Response:**
```json
{
  "task_id": "task_12345",
  "status": "completed",
  "image_url": "/outputs/images/generated_12345.png",
  "progress": 100
}
```

#### List Available Models

```http
GET /text-to-image/models
```

**Response:**
```json
[
  {
    "id": "stable-diffusion-v1-5",
    "name": "Stable Diffusion v1.5",
    "description": "General purpose image generation"
  },
  {
    "id": "stable-diffusion-xl",
    "name": "Stable Diffusion XL",
    "description": "High-resolution image generation"
  }
]
```

---

### Image to Image

#### Transform Image

```http
POST /image-to-image/generate
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Source image |
| prompt | string | Yes | Transformation prompt |
| negative_prompt | string | No | Negative prompt |
| strength | float | No | Transformation strength (0.0-1.0) |
| num_inference_steps | int | No | Number of steps |
| guidance_scale | float | No | Guidance scale |

**Response:**
```json
{
  "task_id": "task_12346",
  "status": "completed",
  "image_url": "/outputs/images/transformed_12346.png"
}
```

---

### Video Generation

#### Text to Video

```http
POST /text-to-video/generate
```

**Request Body:**
```json
{
  "prompt": "a cat walking in a garden",
  "num_frames": 24,
  "fps": 8,
  "num_inference_steps": 25
}
```

#### Image to Video

```http
POST /image-to-video/generate
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Source image |
| num_frames | int | No | Number of frames (default: 24) |
| fps | int | No | Frames per second (default: 8) |

---

## Recognition APIs

### Image Recognition

#### Classify Image

```http
POST /image-recognition/classify
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Image to classify |
| model | string | No | Model ID |

**Response:**
```json
{
  "classifications": [
    {"label": "cat", "confidence": 0.95},
    {"label": "pet", "confidence": 0.89}
  ]
}
```

#### Detect Objects

```http
POST /image-recognition/detect
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": [100, 50, 200, 300]
    }
  ]
}
```

---

### Face Recognition

#### Detect Faces

```http
POST /face-recognition/detect
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "faces": [
    {
      "face_id": "face_001",
      "confidence": 0.98,
      "bbox": [120, 80, 180, 200],
      "landmarks": {
        "left_eye": [145, 120],
        "right_eye": [165, 118]
      }
    }
  ]
}
```

#### Register Face

```http
POST /face-recognition/register
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| image | File | Yes | Face image |
| name | string | Yes | Person's name |

**Response:**
```json
{
  "success": true,
  "face_id": "face_abc123"
}
```

#### Recognize Faces

```http
POST /face-recognition/recognize
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "faces": [
    {
      "face_id": "face_abc123",
      "name": "John Doe",
      "confidence": 0.95,
      "bbox": [100, 50, 150, 200]
    }
  ]
}
```

---

## Audio APIs

### Transcription

#### Transcribe Audio

```http
POST /audio/transcribe
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| audio | File | Yes | Audio file |
| language | string | No | Language code (auto-detect if not specified) |
| model | string | No | Whisper model size |

**Response:**
```json
{
  "task_id": "task_audio_001",
  "status": "completed",
  "text": "Hello, this is a test transcription.",
  "language": "en",
  "duration": 5.2,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is"
    }
  ]
}
```

### Text to Speech

#### Synthesize Speech

```http
POST /audio/synthesize
```

**Request Body:**
```json
{
  "text": "Hello, world!",
  "voice": "default-female",
  "language": "en",
  "speed": 1.0,
  "pitch": 0
}
```

**Response:**
```json
{
  "task_id": "task_tts_001",
  "status": "completed",
  "audio_url": "/outputs/audio/speech_001.mp3",
  "duration": 1.5
}
```

---

## Training APIs

### Model Training

#### Start Training

```http
POST /training/start
```

**Request Body:**
```json
{
  "model_type": "image_classifier",
  "base_model": "resnet50",
  "dataset_path": "/datasets/my_dataset",
  "output_dir": "/models/trained",
  "epochs": 10,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "save_steps": 500
}
```

**Response:**
```json
{
  "task_id": "train_001"
}
```

#### Get Training Status

```http
GET /training/status/{task_id}
```

**Response:**
```json
{
  "task_id": "train_001",
  "status": "running",
  "progress": 45,
  "current_epoch": 5,
  "total_epochs": 10,
  "current_step": 450,
  "total_steps": 1000,
  "loss": 0.0234,
  "learning_rate": 0.00008,
  "metrics": {
    "accuracy": 0.92,
    "f1_score": 0.89
  }
}
```

#### Stop Training

```http
POST /training/stop/{task_id}
```

---

### LLM Fine-tuning

#### Start LoRA Training

```http
POST /llm-finetuning/lora/start
```

**Request Body:**
```json
{
  "base_model": "llama-2-7b",
  "dataset_path": "/datasets/instruction_data.jsonl",
  "output_dir": "/models/lora_adapter",
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.0002,
  "max_length": 512,
  "gradient_accumulation_steps": 4
}
```

#### Start QLoRA Training (4-bit)

```http
POST /llm-finetuning/qlora/start
```

Same parameters as LoRA, with 4-bit quantization enabled.

#### Test Fine-tuned Model

```http
POST /llm-finetuning/test
```

**Request Body:**
```json
{
  "task_id": "lora_001",
  "prompt": "What is machine learning?",
  "max_tokens": 256
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of artificial intelligence..."
}
```

#### Merge Adapter

```http
POST /llm-finetuning/merge
```

**Request Body:**
```json
{
  "task_id": "lora_001",
  "output_path": "/models/merged_model"
}
```

---

## WebSocket Endpoints

### Training Progress

```
ws://localhost:8000/ws/training
```

**Message Types:**

```json
// Progress update
{
  "type": "progress",
  "data": {
    "task_id": "train_001",
    "progress": 45,
    "metrics": {"loss": 0.023}
  }
}

// Status update
{
  "type": "status",
  "data": {
    "task_id": "train_001",
    "status": "completed"
  }
}

// Error
{
  "type": "error",
  "data": {
    "task_id": "train_001",
    "message": "Out of memory"
  }
}
```

### Camera Stream

```
ws://localhost:8000/ws/camera
```

Used for real-time face recognition and object detection with webcam.

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

## Rate Limiting

Default rate limits:
- 100 requests per minute for generation endpoints
- 1000 requests per minute for status endpoints

Custom limits can be configured in the backend settings.
