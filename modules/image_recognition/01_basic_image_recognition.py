"""
基础示例: 图片识别（最简单版本）
使用预训练模型进行图片分类和目标检测
"""

import torch
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import (
    load_image, get_device,
    load_transformers_model_with_fallback,
    load_yolo_model_with_fallback
)
from transformers import AutoImageProcessor, AutoModelForImageClassification
from config.modules_config import (
    DEFAULT_IMAGE_RECOGNITION_MODEL,
    DEFAULT_OBJECT_DETECTION_MODEL,
    LOCAL_IMAGE_RECOGNITION_MODEL_PATH,
    LOCAL_OBJECT_DETECTION_MODEL_PATH
)
import torch.nn.functional as F


def classify_image(
        image_path: str,
        top_k: int = 5,
        local_model_path: str = None
):
    """
    对图片进行分类识别
    
    Args:
        image_path: 图片路径
        top_k: 返回前k个最可能的类别
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则从配置文件读取
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则使用指定的路径
    
    Returns:
        分类结果列表，每个元素包含 (类别名称, 置信度)
    """
    print(f"\n开始识别图片...")
    print(f"图片路径: {image_path}")

    # 获取设备
    device = get_device()

    # 加载图片
    image = load_image(image_path)
    print(f"图片尺寸: {image.size}")

    # 确定本地模型路径的优先级
    if local_model_path is not None:
        model_path = local_model_path if local_model_path else None
    else:
        model_path = LOCAL_IMAGE_RECOGNITION_MODEL_PATH if LOCAL_IMAGE_RECOGNITION_MODEL_PATH else None

    # 加载模型和处理器（优先使用本地模型）
    print("\n正在加载模型...")
    if model_path:
        print(f"本地模型路径: {model_path}")
    else:
        print("本地模型: 已禁用（仅使用在线模型）")

    processor, model = load_transformers_model_with_fallback(
        AutoImageProcessor,
        AutoModelForImageClassification,
        DEFAULT_IMAGE_RECOGNITION_MODEL,
        model_path
    )
    model = model.to(device)
    model.eval()

    print("模型加载完成！")

    # 预处理图片
    inputs = processor(image, return_tensors="pt").to(device)

    # 进行推理
    print("\n正在识别图片...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 获取概率
    probs = F.softmax(logits, dim=-1)

    # 获取top-k结果
    top_probs, top_indices = torch.topk(probs, top_k)

    # 获取类别标签
    id2label = model.config.id2label

    results = []
    print(f"\n识别结果（Top {top_k}）:")
    print("-" * 60)
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0]), 1):
        label = id2label[idx.item()]
        confidence = prob.item() * 100
        results.append((label, confidence))
        print(f"{i}. {label}: {confidence:.2f}%")

    print("-" * 60)
    return results


def detect_objects(
        image_path: str,
        confidence_threshold: float = 0.5,
        local_model_path: str = None
):
    """
    对图片进行目标检测
    
    Args:
        image_path: 图片路径
        confidence_threshold: 置信度阈值
        local_model_path: 本地YOLO模型路径（可选）
                         - 如果为 None，则从配置文件读取
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则使用指定的路径
    
    Returns:
        检测结果列表，每个元素包含 (类别, 置信度, 边界框)
    """
    print(f"\n开始检测图片中的物体...")
    print(f"图片路径: {image_path}")
    print(f"置信度阈值: {confidence_threshold}")

    # 获取设备
    device = get_device()
    device_str = "cuda" if device == "cuda" else "cpu"

    # 确定本地模型路径的优先级
    if local_model_path is not None:
        model_path = local_model_path if local_model_path else None
    else:
        model_path = LOCAL_OBJECT_DETECTION_MODEL_PATH if LOCAL_OBJECT_DETECTION_MODEL_PATH else None

    # 加载模型（优先使用本地模型）
    print("\n正在加载YOLO模型...")
    if model_path:
        print(f"本地模型路径: {model_path}")
    else:
        print("本地模型: 已禁用（仅使用在线模型）")

    model = load_yolo_model_with_fallback(
        DEFAULT_OBJECT_DETECTION_MODEL,
        model_path
    )
    print("模型加载完成！")

    # 进行检测
    print("\n正在检测物体...")
    results = model(image_path, conf=confidence_threshold, device=device_str)

    # results[0].show()
    import uuid

    # 生成一个新的随机 UUID
    uuid_value = str(uuid.uuid4())

    results[0].save(filename=uuid_value + ".jpg")  # 保存到文件

    # 解析结果
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            box = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            label = model.names[cls]

            detections.append({
                "label": label,
                "confidence": conf * 100,
                "bbox": box
            })

    # 打印结果
    print(f"\n检测到 {len(detections)} 个物体:")
    print("-" * 60)
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['label']}: {det['confidence']:.2f}%")
        print(f"   位置: ({det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}) -> ({det['bbox'][2]:.1f}, {det['bbox'][3]:.1f})")
    print("-" * 60)

    return detections


if __name__ == "__main__":
    # 示例1: 图片分类
    # print("=" * 60)
    # print("示例1: 图片分类")
    # print("=" * 60)
    # image_path = "E:\\GIT_AI\\aipicture\\outputs\\images\\advanced_text_to_image\\lion_basic.png"  # 替换为你的图片路径
    # classify_image(image_path, top_k=5)

    # 示例2: 目标检测
    print("\n" + "=" * 60)
    print("示例2: 目标检测")
    print("=" * 60)
    image_path = "E:\\GIT_AI\\aipicture\\modules\\image_recognition\\outputs\\images\\u.png"  # 替换为你的图片路径
    detect_objects(image_path, confidence_threshold=0.5)

    print("\n请先准备一张测试图片，然后取消注释上面的代码并运行")
