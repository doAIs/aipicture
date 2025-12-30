"""
进阶示例: 图片识别（带参数控制）
包含更多识别功能和参数选项
"""

import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import load_image, get_device, set_seed
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
from typing import List, Dict, Tuple


class AdvancedImageRecognition:
    """高级图片识别类"""
    
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
        print(f"初始化图片识别器")
    
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
        image_path: str,
        top_k: int = 5,
        return_probs: bool = False
    ) -> List[Tuple[str, float]]:
        """
        对图片进行分类
        
        Args:
            image_path: 图片路径
            top_k: 返回前k个最可能的类别
            return_probs: 是否返回所有概率
        
        Returns:
            分类结果列表
        """
        self.load_classification_model()
        
        image = load_image(image_path)
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        id2label = self.classification_model.config.id2label
        results = []
        
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = id2label[idx.item()]
            confidence = prob.item() * 100
            results.append((label, confidence))
        
        return results
    
    def detect(
        self,
        image_path: str,
        confidence_threshold: float = 0.5,
        save_result: bool = False,
        output_path: str = None
    ) -> List[Dict]:
        """
        对图片进行目标检测
        
        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值
            save_result: 是否保存标注结果图片
            output_path: 输出图片路径
        
        Returns:
            检测结果列表
        """
        self.load_detection_model()
        
        device_str = "cuda" if self.device == "cuda" else "cpu"
        results = self.detection_model(
            image_path,
            conf=confidence_threshold,
            device=device_str
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                box = boxes.xyxy[i].tolist()
                label = self.detection_model.names[cls]
                
                detections.append({
                    "label": label,
                    "confidence": conf * 100,
                    "bbox": box,
                    "class_id": cls
                })
        
        # 保存标注结果
        if save_result and len(results) > 0:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = f"outputs/images/detection_{base_name}.jpg"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            annotated_image = results[0].plot()
            Image.fromarray(annotated_image).save(output_path)
            print(f"标注结果已保存到: {output_path}")
        
        return detections
    
    def batch_classify(self, image_paths: List[str], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        批量分类多张图片
        
        Args:
            image_paths: 图片路径列表
            top_k: 返回前k个最可能的类别
        
        Returns:
            每张图片的分类结果字典
        """
        results = {}
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n处理第 {i}/{len(image_paths)} 张图片: {image_path}")
            results[image_path] = self.classify(image_path, top_k=top_k)
        return results
    
    def batch_detect(self, image_paths: List[str], confidence_threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """
        批量检测多张图片
        
        Args:
            image_paths: 图片路径列表
            confidence_threshold: 置信度阈值
        
        Returns:
            每张图片的检测结果字典
        """
        results = {}
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n处理第 {i}/{len(image_paths)} 张图片: {image_path}")
            results[image_path] = self.detect(image_path, confidence_threshold=confidence_threshold)
        return results


def main():
    """主函数 - 演示不同功能"""
    recognizer = AdvancedImageRecognition()
    
    # 示例1: 图片分类
    print("\n【示例1】图片分类")
    # image_path = "examples/images/test.jpg"
    # results = recognizer.classify(image_path, top_k=5)
    # print("\n分类结果:")
    # for label, confidence in results:
    #     print(f"  {label}: {confidence:.2f}%")
    
    # 示例2: 目标检测
    print("\n【示例2】目标检测")
    # image_path = "examples/images/test.jpg"
    # detections = recognizer.detect(
    #     image_path,
    #     confidence_threshold=0.5,
    #     save_result=True
    # )
    # print(f"\n检测到 {len(detections)} 个物体:")
    # for det in detections:
    #     print(f"  {det['label']}: {det['confidence']:.2f}%")
    
    # 示例3: 批量处理
    print("\n【示例3】批量处理")
    # image_paths = [
    #     "examples/images/test1.jpg",
    #     "examples/images/test2.jpg"
    # ]
    # batch_results = recognizer.batch_classify(image_paths, top_k=3)
    # for path, results in batch_results.items():
    #     print(f"\n{path}:")
    #     for label, confidence in results:
    #         print(f"  {label}: {confidence:.2f}%")
    
    print("\n请准备测试图片，然后取消注释上面的代码并运行")


if __name__ == "__main__":
    main()

