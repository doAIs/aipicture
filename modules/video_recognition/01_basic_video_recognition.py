"""
基础示例: 视频识别（最简单版本）
对视频进行逐帧识别，包括分类和目标检测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import (
    load_video, get_device, get_video_info,
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
import torch
import torch.nn.functional as F
from typing import List, Dict
from collections import Counter


def classify_video(
    video_path: str, 
    sample_frames: int = 10, 
    top_k: int = 5,
    local_model_path: str = None
):
    """
    对视频进行分类识别
    通过采样帧进行识别，然后汇总结果
    
    Args:
        video_path: 视频路径
        sample_frames: 采样帧数
        top_k: 返回前k个最可能的类别
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则从配置文件读取
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则使用指定的路径
    
    Returns:
        分类结果列表
    """
    print(f"\n开始识别视频...")
    print(f"视频路径: {video_path}")
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    print(f"视频信息: {video_info['frames']} 帧, {video_info['fps']:.2f} fps")
    
    # 获取设备
    device = get_device()
    
    # 加载视频
    print("\n正在加载视频...")
    frames = load_video(video_path)
    
    # 采样帧
    import numpy as np
    if len(frames) > sample_frames:
        indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames
    
    print(f"采样 {len(sampled_frames)} 帧进行分析")
    
    # 确定本地模型路径的优先级
    if local_model_path is not None:
        model_path = local_model_path if local_model_path else None
    else:
        model_path = LOCAL_IMAGE_RECOGNITION_MODEL_PATH if LOCAL_IMAGE_RECOGNITION_MODEL_PATH else None
    
    # 加载模型（优先使用本地模型）
    print("\n正在加载分类模型...")
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
    
    # 对每帧进行分类
    print("\n正在分析视频帧...")
    all_predictions = []
    
    with torch.no_grad():
        for i, frame in enumerate(sampled_frames, 1):
            print(f"处理第 {i}/{len(sampled_frames)} 帧...")
            inputs = processor(frame, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # 获取top-1预测
            top_prob, top_idx = torch.topk(probs, 1)
            label = model.config.id2label[top_idx[0].item()]
            confidence = top_prob[0].item() * 100
            all_predictions.append((label, confidence))
    
    # 汇总结果
    print("\n汇总识别结果...")
    label_counts = Counter([label for label, _ in all_predictions])
    
    # 计算平均置信度
    label_confidences = {}
    for label, conf in all_predictions:
        if label not in label_confidences:
            label_confidences[label] = []
        label_confidences[label].append(conf)
    
    # 生成结果
    results = []
    for label, count in label_counts.most_common(top_k):
        avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
        percentage = (count / len(sampled_frames)) * 100
        results.append({
            "label": label,
            "frequency": count,
            "percentage": percentage,
            "avg_confidence": avg_conf
        })
    
    # 打印结果
    print(f"\n视频识别结果（Top {top_k}）:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label']}")
        print(f"   出现频率: {result['frequency']}/{len(sampled_frames)} ({result['percentage']:.1f}%)")
        print(f"   平均置信度: {result['avg_confidence']:.2f}%")
    print("-" * 60)
    
    return results


def detect_objects_in_video(
    video_path: str,
    sample_frames: int = 10,
    confidence_threshold: float = 0.5,
    local_model_path: str = None
):
    """
    对视频进行目标检测
    通过采样帧进行检测，然后汇总结果
    
    Args:
        video_path: 视频路径
        sample_frames: 采样帧数
        confidence_threshold: 置信度阈值
        local_model_path: 本地YOLO模型路径（可选）
                         - 如果为 None，则从配置文件读取
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则使用指定的路径
    
    Returns:
        检测结果字典
    """
    print(f"\n开始检测视频中的物体...")
    print(f"视频路径: {video_path}")
    print(f"置信度阈值: {confidence_threshold}")
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    print(f"视频信息: {video_info['frames']} 帧, {video_info['fps']:.2f} fps")
    
    # 获取设备
    device = get_device()
    device_str = "cuda" if device == "cuda" else "cpu"
    
    # 加载视频
    print("\n正在加载视频...")
    frames = load_video(video_path)
    
    # 采样帧
    import numpy as np
    if len(frames) > sample_frames:
        indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
    else:
        sampled_frames = frames
    
    print(f"采样 {len(sampled_frames)} 帧进行分析")
    
    # 确定本地模型路径的优先级
    if local_model_path is not None:
        model_path = local_model_path if local_model_path else None
    else:
        model_path = LOCAL_OBJECT_DETECTION_MODEL_PATH if LOCAL_OBJECT_DETECTION_MODEL_PATH else None
    
    # 加载模型（优先使用本地模型）
    print("\n正在加载检测模型...")
    if model_path:
        print(f"本地模型路径: {model_path}")
    else:
        print("本地模型: 已禁用（仅使用在线模型）")
    
    model = load_yolo_model_with_fallback(
        DEFAULT_OBJECT_DETECTION_MODEL,
        model_path
    )
    print("模型加载完成！")
    
    # 对每帧进行检测
    print("\n正在分析视频帧...")
    all_detections = []
    
    for i, frame in enumerate(sampled_frames, 1):
        print(f"处理第 {i}/{len(sampled_frames)} 帧...")
        results = model(frame, conf=confidence_threshold, device=device_str)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for j in range(len(boxes)):
                cls = int(boxes.cls[j])
                conf = float(boxes.conf[j])
                label = model.names[cls]
                all_detections.append({
                    "label": label,
                    "confidence": conf * 100,
                    "frame": i
                })
    
    # 汇总结果
    print("\n汇总检测结果...")
    from collections import Counter
    label_counts = Counter([det["label"] for det in all_detections])
    
    # 计算平均置信度
    label_confidences = {}
    for det in all_detections:
        label = det["label"]
        if label not in label_confidences:
            label_confidences[label] = []
        label_confidences[label].append(det["confidence"])
    
    # 生成结果
    results = []
    for label, count in label_counts.most_common():
        avg_conf = sum(label_confidences[label]) / len(label_confidences[label])
        percentage = (count / len(sampled_frames)) * 100
        results.append({
            "label": label,
            "detections": count,
            "percentage": percentage,
            "avg_confidence": avg_conf
        })
    
    # 打印结果
    print(f"\n视频检测结果:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label']}")
        print(f"   检测次数: {result['detections']} (出现在 {result['percentage']:.1f}% 的帧中)")
        print(f"   平均置信度: {result['avg_confidence']:.2f}%")
    print("-" * 60)
    
    return results


if __name__ == "__main__":
    # 示例1: 视频分类
    print("=" * 60)
    print("示例1: 视频分类")
    print("=" * 60)
    # video_path = "examples/videos/test.mp4"  # 替换为你的视频路径
    # classify_video(video_path, sample_frames=10, top_k=5)
    
    # 示例2: 视频目标检测
    print("\n" + "=" * 60)
    print("示例2: 视频目标检测")
    print("=" * 60)
    # video_path = "examples/videos/test.mp4"  # 替换为你的视频路径
    # detect_objects_in_video(video_path, sample_frames=10, confidence_threshold=0.5)
    
    print("\n请先准备一个测试视频，然后取消注释上面的代码并运行")

