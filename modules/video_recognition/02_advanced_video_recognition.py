"""
进阶示例: 视频识别（带参数控制）
包含更多识别功能和参数选项
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import load_video, get_device, get_video_info
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from typing import List, Dict
from collections import Counter
import numpy as np


class AdvancedVideoRecognition:
    """高级视频识别类"""
    
    def __init__(self, classification_model: str = "google/vit-base-patch16-224"):
        """
        初始化识别器
        
        Args:
            classification_model: 分类模型名称
        """
        self.device = get_device()
        self.classification_model_name = classification_model
        self.processor = None
        self.classification_model = None
        self.detection_model = None
        print(f"初始化视频识别器")
    
    def load_classification_model(self):
        """加载分类模型（延迟加载）"""
        if self.classification_model is None:
            print("\n正在加载分类模型（首次运行需要下载，请耐心等待）...")
            self.processor = AutoImageProcessor.from_pretrained(self.classification_model_name)
            self.classification_model = AutoModelForImageClassification.from_pretrained(
                self.classification_model_name
            )
            self.classification_model = self.classification_model.to(self.device)
            self.classification_model.eval()
            print("分类模型加载完成！")
    
    def load_detection_model(self):
        """加载检测模型（延迟加载）"""
        if self.detection_model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("请安装 ultralytics: pip install ultralytics")
            
            print("\n正在加载检测模型（首次运行需要下载，请耐心等待）...")
            self.detection_model = YOLO("yolov8n.pt")
            print("检测模型加载完成！")
    
    def classify(
        self,
        video_path: str,
        sample_frames: int = 10,
        top_k: int = 5,
        sampling_strategy: str = "uniform"
    ) -> List[Dict]:
        """
        对视频进行分类
        
        Args:
            video_path: 视频路径
            sample_frames: 采样帧数
            top_k: 返回前k个最可能的类别
            sampling_strategy: 采样策略 ("uniform", "random", "all")
        
        Returns:
            分类结果列表
        """
        self.load_classification_model()
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        print(f"视频: {video_info['frames']} 帧, {video_info['fps']:.2f} fps")
        
        # 加载视频
        frames = load_video(video_path)
        
        # 采样帧
        if sampling_strategy == "all":
            sampled_frames = frames
        elif sampling_strategy == "random":
            if len(frames) > sample_frames:
                indices = np.random.choice(len(frames), sample_frames, replace=False)
                sampled_frames = [frames[i] for i in sorted(indices)]
            else:
                sampled_frames = frames
        else:  # uniform
            if len(frames) > sample_frames:
                indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = frames
        
        print(f"采样 {len(sampled_frames)} 帧进行分析")
        
        # 对每帧进行分类
        all_predictions = []
        with torch.no_grad():
            for i, frame in enumerate(sampled_frames, 1):
                if i % 5 == 0:
                    print(f"处理第 {i}/{len(sampled_frames)} 帧...")
                inputs = self.processor(frame, return_tensors="pt").to(self.device)
                outputs = self.classification_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
                top_prob, top_idx = torch.topk(probs, 1)
                label = self.classification_model.config.id2label[top_idx[0].item()]
                confidence = top_prob[0].item() * 100
                all_predictions.append((label, confidence))
        
        # 汇总结果
        label_counts = Counter([label for label, _ in all_predictions])
        label_confidences = {}
        for label, conf in all_predictions:
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)
        
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
        
        return results
    
    def detect(
        self,
        video_path: str,
        sample_frames: int = 10,
        confidence_threshold: float = 0.5,
        sampling_strategy: str = "uniform",
        save_annotated_frames: bool = False,
        output_dir: str = None
    ) -> List[Dict]:
        """
        对视频进行目标检测
        
        Args:
            video_path: 视频路径
            sample_frames: 采样帧数
            confidence_threshold: 置信度阈值
            sampling_strategy: 采样策略
            save_annotated_frames: 是否保存标注帧
            output_dir: 输出目录
        
        Returns:
            检测结果列表
        """
        self.load_detection_model()
        
        # 获取视频信息
        video_info = get_video_info(video_path)
        print(f"视频: {video_info['frames']} 帧, {video_info['fps']:.2f} fps")
        
        # 加载视频
        frames = load_video(video_path)
        
        # 采样帧
        if sampling_strategy == "all":
            sampled_frames = frames
        elif sampling_strategy == "random":
            if len(frames) > sample_frames:
                indices = np.random.choice(len(frames), sample_frames, replace=False)
                sampled_frames = [frames[i] for i in sorted(indices)]
            else:
                sampled_frames = frames
        else:  # uniform
            if len(frames) > sample_frames:
                indices = np.linspace(0, len(frames) - 1, sample_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            else:
                sampled_frames = frames
        
        print(f"采样 {len(sampled_frames)} 帧进行分析")
        
        device_str = "cuda" if self.device == "cuda" else "cpu"
        
        # 对每帧进行检测
        all_detections = []
        for i, frame in enumerate(sampled_frames, 1):
            if i % 5 == 0:
                print(f"处理第 {i}/{len(sampled_frames)} 帧...")
            
            results = self.detection_model(frame, conf=confidence_threshold, device=device_str)
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                frame_detections = []
                for j in range(len(boxes)):
                    cls = int(boxes.cls[j])
                    conf = float(boxes.conf[j])
                    box = boxes.xyxy[j].tolist()
                    label = self.detection_model.names[cls]
                    
                    frame_detections.append({
                        "label": label,
                        "confidence": conf * 100,
                        "bbox": box,
                        "class_id": cls
                    })
                
                all_detections.extend(frame_detections)
                
                # 保存标注帧
                if save_annotated_frames:
                    if output_dir is None:
                        output_dir = "outputs/images/video_detection"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    annotated_image = results[0].plot()
                    from PIL import Image
                    Image.fromarray(annotated_image).save(
                        os.path.join(output_dir, f"frame_{i:04d}.jpg")
                    )
        
        # 汇总结果
        label_counts = Counter([det["label"] for det in all_detections])
        label_confidences = {}
        for det in all_detections:
            label = det["label"]
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(det["confidence"])
        
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
        
        if save_annotated_frames:
            print(f"\n标注帧已保存到: {output_dir}")
        
        return results


def main():
    """主函数 - 演示不同功能"""
    recognizer = AdvancedVideoRecognition()
    
    # 示例1: 视频分类
    print("\n【示例1】视频分类")
    # video_path = "examples/videos/test.mp4"
    # results = recognizer.classify(
    #     video_path,
    #     sample_frames=20,
    #     top_k=5,
    #     sampling_strategy="uniform"
    # )
    # print("\n分类结果:")
    # for result in results:
    #     print(f"  {result['label']}: {result['percentage']:.1f}% (置信度: {result['avg_confidence']:.2f}%)")
    
    # 示例2: 视频目标检测
    print("\n【示例2】视频目标检测")
    # video_path = "examples/videos/test.mp4"
    # results = recognizer.detect(
    #     video_path,
    #     sample_frames=20,
    #     confidence_threshold=0.5,
    #     save_annotated_frames=True
    # )
    # print("\n检测结果:")
    # for result in results:
    #     print(f"  {result['label']}: {result['detections']} 次检测 (出现在 {result['percentage']:.1f}% 的帧中)")
    
    print("\n请准备测试视频，然后取消注释上面的代码并运行")


if __name__ == "__main__":
    main()

