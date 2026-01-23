# Training Guide

**English** | [中文](TRAINING_GUIDE_CN.md)

Comprehensive guide for training and fine-tuning AI models on the platform.

## Table of Contents

1. [Overview](#overview)
2. [Image Classifier Training](#image-classifier-training)
3. [Object Detector Training](#object-detector-training)
4. [LLM Fine-tuning with LoRA](#llm-fine-tuning-with-lora)
5. [QLoRA Fine-tuning](#qlora-fine-tuning)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Multimedia Platform supports training several types of models:

| Model Type | Use Case | Typical Training Time |
|------------|----------|----------------------|
| Image Classifier | Categorize images | 1-4 hours |
| Object Detector | Detect objects in images | 4-12 hours |
| Text-to-Image LoRA | Customize image generation | 2-8 hours |
| LLM LoRA | Fine-tune language models | 4-24 hours |

### Hardware Requirements

| Training Type | Minimum VRAM | Recommended VRAM |
|--------------|--------------|------------------|
| Image Classifier | 4GB | 8GB+ |
| Object Detector | 8GB | 12GB+ |
| SD LoRA | 8GB | 12GB+ |
| LLM LoRA (7B) | 16GB | 24GB+ |
| LLM QLoRA (7B) | 8GB | 12GB+ |

---

## Image Classifier Training

### Dataset Preparation

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   └── ...
│   └── class_n/
│       └── ...
└── val/
    ├── class_1/
    │   └── ...
    └── class_n/
        └── ...
```

### Recommended Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | ResNet-50 | Good balance of speed and accuracy |
| Epochs | 10-20 | Depends on dataset size |
| Batch Size | 8-32 | Adjust based on VRAM |
| Learning Rate | 1e-4 to 1e-3 | Start with 1e-4 |
| Image Size | 224x224 | Standard for most models |

### Training via UI

1. Navigate to **Model Training** page
2. Select **Image Classifier** as model type
3. Choose a base model (ResNet-50 recommended)
4. Upload your dataset (ZIP file)
5. Configure training parameters
6. Click **Start Training**

### Training via API

```python
import requests

response = requests.post("http://localhost:8000/api/training/start", json={
    "model_type": "image_classifier",
    "base_model": "resnet50",
    "dataset_path": "/datasets/my_images",
    "epochs": 15,
    "batch_size": 16,
    "learning_rate": 0.0001
})

task_id = response.json()["task_id"]
```

---

## Object Detector Training

### Dataset Format

YOLO format dataset structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt  # Same name as image
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

**Label format (YOLO):**
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

**data.yaml:**
```yaml
train: /path/to/images/train
val: /path/to/images/val
nc: 3  # number of classes
names: ['cat', 'dog', 'bird']
```

### Recommended Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | YOLOv8n | Fast, good for most cases |
| Epochs | 50-100 | Object detection needs more |
| Batch Size | 16 | Adjust based on VRAM |
| Image Size | 640 | Standard YOLO input |

---

## LLM Fine-tuning with LoRA

LoRA (Low-Rank Adaptation) enables efficient fine-tuning of large language models.

### Dataset Format

JSONL format with instruction-response pairs:

```jsonl
{"instruction": "Translate to French: Hello", "response": "Bonjour"}
{"instruction": "Summarize: ...", "response": "..."}
```

Or conversation format:

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### LoRA Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| lora_r | 8-64 | Rank of adaptation matrices |
| lora_alpha | 16-128 | Scaling factor (usually 2x lora_r) |
| lora_dropout | 0.05-0.1 | Dropout for regularization |
| target_modules | q_proj, v_proj | Which layers to adapt |

### Recommended Settings by Model Size

**7B Models (LLaMA 2, Mistral):**
```json
{
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "learning_rate": 2e-4,
  "epochs": 3
}
```

**13B Models:**
```json
{
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "batch_size": 2,
  "gradient_accumulation_steps": 8,
  "learning_rate": 1e-4,
  "epochs": 3
}
```

### Training via UI

1. Navigate to **LLM Fine-tuning** page
2. Select base model (e.g., LLaMA 2 7B)
3. Choose **LoRA** method
4. Upload JSONL training data
5. Configure LoRA parameters
6. Click **Start Fine-tuning**

---

## QLoRA Fine-tuning

QLoRA uses 4-bit quantization to reduce memory usage significantly.

### Memory Comparison

| Model | Full Fine-tuning | LoRA | QLoRA |
|-------|-----------------|------|-------|
| 7B | 28GB+ | 16GB | 6GB |
| 13B | 52GB+ | 32GB | 10GB |
| 70B | 280GB+ | 160GB | 48GB |

### When to Use QLoRA

- Limited VRAM (< 16GB)
- Training 13B+ models on consumer hardware
- Memory-constrained cloud instances

### QLoRA-Specific Settings

```json
{
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4",
  "use_nested_quant": false
}
```

---

## Best Practices

### Data Quality

1. **Clean your data**: Remove duplicates, errors, and low-quality samples
2. **Balance classes**: Equal representation improves results
3. **Augmentation**: Use data augmentation for small datasets
4. **Validation split**: Keep 10-20% for validation

### Training Tips

1. **Start small**: Train on a subset first to verify setup
2. **Monitor loss**: Watch for overfitting (val loss increasing)
3. **Learning rate**: Use warmup and cosine decay
4. **Checkpoints**: Save regularly to resume training

### Resource Management

1. **Gradient accumulation**: Simulate larger batches
2. **Mixed precision**: Use fp16 for faster training
3. **Gradient checkpointing**: Trade compute for memory

### Evaluation

1. **Hold-out test set**: Final evaluation on unseen data
2. **Multiple metrics**: Don't rely on a single metric
3. **Qualitative review**: Manually inspect outputs

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use QLoRA instead of LoRA
4. Reduce sequence length (for LLMs)
5. Use smaller base model

### Training Loss Not Decreasing

**Possible causes:**
1. Learning rate too low → Increase
2. Learning rate too high → Decrease
3. Bad data quality → Clean dataset
4. Bug in data preprocessing → Verify data

### Overfitting

**Signs:** Training loss decreases but validation loss increases

**Solutions:**
1. Add more training data
2. Use data augmentation
3. Increase dropout
4. Reduce model complexity
5. Use early stopping

### Slow Training

**Optimization:**
1. Enable GPU acceleration
2. Use mixed precision (fp16)
3. Optimize data loading (more workers)
4. Use SSD for dataset storage

---

## Exporting Models

### Export LoRA Adapter

Adapters are small (< 100MB) and can be shared easily:

```bash
# Via UI: Click "Export LoRA Adapter" button
# Via API:
curl -X GET "http://localhost:8000/api/llm-finetuning/export/{task_id}"
```

### Merge and Export Full Model

Creates a standalone model with adapter merged:

```bash
# Via UI: Click "Merge & Export Full Model" button
# Via API:
curl -X POST "http://localhost:8000/api/llm-finetuning/merge" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "...", "output_path": "/models/merged"}'
```

---

## Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
