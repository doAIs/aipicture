# 训练指南

平台 AI 模型训练和微调完整指南。

[English](TRAINING_GUIDE.md) | **中文**

## 目录

1. [概述](#概述)
2. [图像分类器训练](#图像分类器训练)
3. [目标检测器训练](#目标检测器训练)
4. [LLM LoRA 微调](#llm-lora-微调)
5. [QLoRA 微调](#qlora-微调)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

---

## 概述

AI 多媒体平台支持训练多种类型的模型：

| 模型类型 | 用途 | 典型训练时间 |
|----------|------|--------------|
| 图像分类器 | 图像分类 | 1-4 小时 |
| 目标检测器 | 检测图像中的物体 | 4-12 小时 |
| 文生图 LoRA | 自定义图像生成 | 2-8 小时 |
| LLM LoRA | 微调大语言模型 | 4-24 小时 |

### 硬件要求

| 训练类型 | 最低显存 | 推荐显存 |
|----------|----------|----------|
| 图像分类器 | 4GB | 8GB+ |
| 目标检测器 | 8GB | 12GB+ |
| SD LoRA | 8GB | 12GB+ |
| LLM LoRA (7B) | 16GB | 24GB+ |
| LLM QLoRA (7B) | 8GB | 12GB+ |

---

## 图像分类器训练

### 数据集准备

按以下结构组织数据集：

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

### 推荐设置

| 参数 | 值 | 描述 |
|------|----|----|
| 基础模型 | ResNet-50 | 速度和精度的良好平衡 |
| 轮次 | 10-20 | 取决于数据集大小 |
| 批次大小 | 8-32 | 根据显存调整 |
| 学习率 | 1e-4 到 1e-3 | 建议从 1e-4 开始 |
| 图像尺寸 | 224x224 | 大多数模型的标准尺寸 |

### 通过界面训练

1. 导航到 **模型训练** 页面
2. 选择 **图像分类器** 作为模型类型
3. 选择基础模型（推荐 ResNet-50）
4. 上传数据集（ZIP 文件）
5. 配置训练参数
6. 点击 **开始训练**

### 通过 API 训练

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

## 目标检测器训练

### 数据集格式

YOLO 格式数据集结构：

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
│   │   ├── img001.txt  # 与图像同名
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

**标注格式（YOLO）：**
```
class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

**data.yaml：**
```yaml
train: /path/to/images/train
val: /path/to/images/val
nc: 3  # 类别数量
names: ['猫', '狗', '鸟']
```

### 推荐设置

| 参数 | 值 | 描述 |
|------|----|----|
| 基础模型 | YOLOv8n | 快速，适用于大多数场景 |
| 轮次 | 50-100 | 目标检测需要更多轮次 |
| 批次大小 | 16 | 根据显存调整 |
| 图像尺寸 | 640 | YOLO 标准输入 |

---

## LLM LoRA 微调

LoRA（低秩适配）可以高效地微调大型语言模型。

### 数据集格式

指令-响应对的 JSONL 格式：

```jsonl
{"instruction": "翻译成英文：你好", "response": "Hello"}
{"instruction": "总结：...", "response": "..."}
```

或对话格式：

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### LoRA 参数

| 参数 | 典型值 | 描述 |
|------|--------|------|
| lora_r | 8-64 | 适配矩阵的秩 |
| lora_alpha | 16-128 | 缩放因子（通常为 lora_r 的 2 倍） |
| lora_dropout | 0.05-0.1 | 正则化的 Dropout |
| target_modules | q_proj, v_proj | 要适配的层 |

### 按模型大小推荐设置

**7B 模型（LLaMA 2, Mistral）：**
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

**13B 模型：**
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

### 通过界面训练

1. 导航到 **LLM 微调** 页面
2. 选择基础模型（如 LLaMA 2 7B）
3. 选择 **LoRA** 方法
4. 上传 JSONL 训练数据
5. 配置 LoRA 参数
6. 点击 **开始微调**

---

## QLoRA 微调

QLoRA 使用 4 位量化显著减少内存使用。

### 内存对比

| 模型 | 全量微调 | LoRA | QLoRA |
|------|----------|------|-------|
| 7B | 28GB+ | 16GB | 6GB |
| 13B | 52GB+ | 32GB | 10GB |
| 70B | 280GB+ | 160GB | 48GB |

### 何时使用 QLoRA

- 显存有限（< 16GB）
- 在消费级硬件上训练 13B+ 模型
- 内存受限的云实例

### QLoRA 特定设置

```json
{
  "use_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_quant_type": "nf4",
  "use_nested_quant": false
}
```

---

## 最佳实践

### 数据质量

1. **清洗数据**: 移除重复、错误和低质量样本
2. **平衡类别**: 各类别均衡分布可改善结果
3. **数据增强**: 对小数据集使用数据增强
4. **验证集划分**: 保留 10-20% 用于验证

### 训练技巧

1. **从小开始**: 先在子集上训练以验证设置
2. **监控损失**: 注意过拟合（验证损失上升）
3. **学习率**: 使用预热和余弦衰减
4. **检查点**: 定期保存以便恢复训练

### 资源管理

1. **梯度累积**: 模拟更大的批次
2. **混合精度**: 使用 fp16 加速训练
3. **梯度检查点**: 用计算换内存

### 评估

1. **保留测试集**: 在未见数据上最终评估
2. **多指标**: 不要只依赖单一指标
3. **定性检查**: 人工检查输出

---

## 故障排除

### 内存不足 (OOM)

**解决方案：**
1. 减小批次大小
2. 启用梯度检查点
3. 使用 QLoRA 代替 LoRA
4. 减小序列长度（对于 LLM）
5. 使用较小的基础模型

### 训练损失不下降

**可能原因：**
1. 学习率太低 → 增加
2. 学习率太高 → 降低
3. 数据质量差 → 清洗数据集
4. 数据预处理错误 → 验证数据

### 过拟合

**表现:** 训练损失下降但验证损失上升

**解决方案：**
1. 增加训练数据
2. 使用数据增强
3. 增加 Dropout
4. 降低模型复杂度
5. 使用早停

### 训练缓慢

**优化方法：**
1. 启用 GPU 加速
2. 使用混合精度 (fp16)
3. 优化数据加载（更多 workers）
4. 使用 SSD 存储数据集

---

## 导出模型

### 导出 LoRA 适配器

适配器文件很小（< 100MB），易于分享：

```bash
# 通过界面: 点击"导出 LoRA 适配器"按钮
# 通过 API:
curl -X GET "http://localhost:8000/api/llm-finetuning/export/{task_id}"
```

### 合并导出完整模型

创建合并了适配器的独立模型：

```bash
# 通过界面: 点击"合并并导出完整模型"按钮
# 通过 API:
curl -X POST "http://localhost:8000/api/llm-finetuning/merge" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "...", "output_path": "/models/merged"}'
```

---

## 参考资源

- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [YOLOv8 文档](https://docs.ultralytics.com)
