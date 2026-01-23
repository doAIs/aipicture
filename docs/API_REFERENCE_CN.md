# API 参考文档

AI 多媒体平台完整 API 文档。

[English](API_REFERENCE.md) | **中文**

## 基础 URL

```
http://localhost:8000/api
```

## 认证

大多数接口不需要认证。对于受保护的接口，需要在请求头中包含授权信息：

```
Authorization: Bearer <token>
```

---

## 生成类 API

### 文生图 (Text to Image)

#### 生成图像

```http
POST /text-to-image/generate
```

**请求体：**
```json
{
  "prompt": "海边美丽的日落",
  "negative_prompt": "模糊, 低质量",
  "width": 512,
  "height": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42,
  "model_id": "stable-diffusion-v1-5"
}
```

**响应：**
```json
{
  "task_id": "task_12345",
  "status": "processing",
  "image_url": null,
  "progress": 0
}
```

#### 获取生成状态

```http
GET /text-to-image/status/{task_id}
```

**响应：**
```json
{
  "task_id": "task_12345",
  "status": "completed",
  "image_url": "/outputs/images/generated_12345.png",
  "progress": 100
}
```

#### 获取可用模型列表

```http
GET /text-to-image/models
```

**响应：**
```json
[
  {
    "id": "stable-diffusion-v1-5",
    "name": "Stable Diffusion v1.5",
    "description": "通用图像生成模型"
  },
  {
    "id": "stable-diffusion-xl",
    "name": "Stable Diffusion XL",
    "description": "高分辨率图像生成模型"
  }
]
```

---

### 图生图 (Image to Image)

#### 图像转换

```http
POST /image-to-image/generate
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| image | File | 是 | 源图像 |
| prompt | string | 是 | 转换提示词 |
| negative_prompt | string | 否 | 负面提示词 |
| strength | float | 否 | 转换强度 (0.0-1.0) |
| num_inference_steps | int | 否 | 推理步数 |
| guidance_scale | float | 否 | 引导系数 |

**响应：**
```json
{
  "task_id": "task_12346",
  "status": "completed",
  "image_url": "/outputs/images/transformed_12346.png"
}
```

---

### 视频生成

#### 文生视频

```http
POST /text-to-video/generate
```

**请求体：**
```json
{
  "prompt": "一只猫在花园里散步",
  "num_frames": 24,
  "fps": 8,
  "num_inference_steps": 25
}
```

#### 图生视频

```http
POST /image-to-video/generate
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| image | File | 是 | 源图像 |
| num_frames | int | 否 | 帧数 (默认: 24) |
| fps | int | 否 | 每秒帧数 (默认: 8) |

---

## 识别类 API

### 图像识别

#### 图像分类

```http
POST /image-recognition/classify
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| image | File | 是 | 待分类图像 |
| model | string | 否 | 模型 ID |

**响应：**
```json
{
  "classifications": [
    {"label": "猫", "confidence": 0.95},
    {"label": "宠物", "confidence": 0.89}
  ]
}
```

#### 目标检测

```http
POST /image-recognition/detect
Content-Type: multipart/form-data
```

**响应：**
```json
{
  "detections": [
    {
      "label": "人",
      "confidence": 0.92,
      "bbox": [100, 50, 200, 300]
    }
  ]
}
```

---

### 人脸识别

#### 检测人脸

```http
POST /face-recognition/detect
Content-Type: multipart/form-data
```

**响应：**
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

#### 注册人脸

```http
POST /face-recognition/register
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| image | File | 是 | 人脸图像 |
| name | string | 是 | 人员姓名 |

**响应：**
```json
{
  "success": true,
  "face_id": "face_abc123"
}
```

#### 识别人脸

```http
POST /face-recognition/recognize
Content-Type: multipart/form-data
```

**响应：**
```json
{
  "faces": [
    {
      "face_id": "face_abc123",
      "name": "张三",
      "confidence": 0.95,
      "bbox": [100, 50, 150, 200]
    }
  ]
}
```

---

## 音频 API

### 语音转录

#### 语音转文字

```http
POST /audio/transcribe
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| audio | File | 是 | 音频文件 |
| language | string | 否 | 语言代码（未指定则自动检测） |
| model | string | 否 | Whisper 模型大小 |

**响应：**
```json
{
  "task_id": "task_audio_001",
  "status": "completed",
  "text": "你好，这是一段测试转录。",
  "language": "zh",
  "duration": 5.2,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "你好，这是"
    }
  ]
}
```

### 文字转语音

#### 语音合成

```http
POST /audio/synthesize
```

**请求体：**
```json
{
  "text": "你好，世界！",
  "voice": "default-female",
  "language": "zh",
  "speed": 1.0,
  "pitch": 0
}
```

**响应：**
```json
{
  "task_id": "task_tts_001",
  "status": "completed",
  "audio_url": "/outputs/audio/speech_001.mp3",
  "duration": 1.5
}
```

---

## 训练 API

### 模型训练

#### 开始训练

```http
POST /training/start
```

**请求体：**
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

**响应：**
```json
{
  "task_id": "train_001"
}
```

#### 获取训练状态

```http
GET /training/status/{task_id}
```

**响应：**
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

#### 停止训练

```http
POST /training/stop/{task_id}
```

---

### LLM 微调

#### 开始 LoRA 训练

```http
POST /llm-finetuning/lora/start
```

**请求体：**
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

#### 开始 QLoRA 训练（4位量化）

```http
POST /llm-finetuning/qlora/start
```

参数与 LoRA 相同，启用 4 位量化。

#### 测试微调模型

```http
POST /llm-finetuning/test
```

**请求体：**
```json
{
  "task_id": "lora_001",
  "prompt": "什么是机器学习？",
  "max_tokens": 256
}
```

**响应：**
```json
{
  "response": "机器学习是人工智能的一个子集..."
}
```

#### 合并适配器

```http
POST /llm-finetuning/merge
```

**请求体：**
```json
{
  "task_id": "lora_001",
  "output_path": "/models/merged_model"
}
```

---

## WebSocket 接口

### 训练进度

```
ws://localhost:8000/ws/training
```

**消息类型：**

```json
// 进度更新
{
  "type": "progress",
  "data": {
    "task_id": "train_001",
    "progress": 45,
    "metrics": {"loss": 0.023}
  }
}

// 状态更新
{
  "type": "status",
  "data": {
    "task_id": "train_001",
    "status": "completed"
  }
}

// 错误
{
  "type": "error",
  "data": {
    "task_id": "train_001",
    "message": "内存不足"
  }
}
```

### 摄像头流

```
ws://localhost:8000/ws/camera
```

用于实时人脸识别和摄像头目标检测。

---

## 错误响应

所有接口返回的错误格式如下：

```json
{
  "detail": "描述错误信息的消息"
}
```

**HTTP 状态码：**
| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 400 | 请求错误 |
| 404 | 未找到 |
| 422 | 验证错误 |
| 500 | 服务器内部错误 |

---

## 速率限制

默认速率限制：
- 生成类接口：每分钟 100 次请求
- 状态查询接口：每分钟 1000 次请求

可在后端设置中自定义限制。
