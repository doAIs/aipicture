"""
综合示例: 完整功能演示
展示所有功能模块的完整工作流程
包括：文生图、文生视频、图生图、图生视频、视频生视频、图片识别、视频识别
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

from modules.text_to_image import AdvancedTextToImage
from modules.image_to_image import AdvancedImageToImage
from modules.text_to_video import AdvancedTextToVideo
from modules.image_to_video import AdvancedImageToVideo
from modules.video_to_video import AdvancedVideoToVideo
from modules.image_recognition import AdvancedImageRecognition
from modules.video_recognition import AdvancedVideoRecognition


class ComprehensiveAIGenerator:
    """综合AI生成器 - 整合所有功能"""
    
    def __init__(self):
        """初始化所有生成器"""
        print("="*60)
        print("初始化综合AI生成器")
        print("="*60)
        
        self.text_to_image_gen = AdvancedTextToImage()
        self.image_to_image_gen = AdvancedImageToImage()
        self.text_to_video_gen = AdvancedTextToVideo()
        self.image_to_video_gen = AdvancedImageToVideo()
        self.video_to_video_gen = AdvancedVideoToVideo()
        self.image_recognizer = AdvancedImageRecognition()
        self.video_recognizer = AdvancedVideoRecognition()
        
        print("\n✅ 所有模块初始化完成！")
    
    def demo_text_to_image(self):
        """演示文本生成图片"""
        print("\n" + "="*60)
        print("功能1: 文本生成图片")
        print("="*60)
        
        prompt = "a beautiful landscape with mountains and lake, sunset, peaceful, photorealistic"
        self.text_to_image_gen.generate(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted",
            output_name="comprehensive_text_to_image"
        )
    
    def demo_image_to_image(self, image_path: str):
        """演示图片生成图片"""
        print("\n" + "="*60)
        print("功能2: 图片生成图片")
        print("="*60)
        
        if not os.path.exists(image_path):
            print(f"⚠️  图片不存在: {image_path}")
            print("跳过此功能演示")
            return
        
        self.image_to_image_gen.generate(
            image_path=image_path,
            prompt="oil painting style, artistic, detailed brushstrokes",
            strength=0.7,
            output_name="comprehensive_image_to_image"
        )
    
    def demo_text_to_video(self):
        """演示文本生成视频"""
        print("\n" + "="*60)
        print("功能3: 文本生成视频")
        print("="*60)
        print("注意：视频生成需要较长时间，请耐心等待...")
        
        prompt = "a beautiful sunset over the ocean, peaceful, serene"
        self.text_to_video_gen.generate(
            prompt=prompt,
            num_frames=16,
            fps=8,
            output_name="comprehensive_text_to_video"
        )
    
    def demo_image_to_video(self, image_path: str):
        """演示图片生成视频"""
        print("\n" + "="*60)
        print("功能4: 图片生成视频")
        print("="*60)
        print("注意：视频生成需要较长时间，请耐心等待...")
        
        if not os.path.exists(image_path):
            print(f"⚠️  图片不存在: {image_path}")
            print("跳过此功能演示")
            return
        
        self.image_to_video_gen.generate(
            image_path=image_path,
            num_frames=14,
            fps=7,
            motion_bucket_id=127,
            output_name="comprehensive_image_to_video"
        )
    
    def demo_video_to_video(self, video_path: str):
        """演示视频生成视频"""
        print("\n" + "="*60)
        print("功能5: 视频生成视频")
        print("="*60)
        print("注意：视频处理需要较长时间，请耐心等待...")
        
        if not os.path.exists(video_path):
            print(f"⚠️  视频不存在: {video_path}")
            print("跳过此功能演示")
            return
        
        self.video_to_video_gen.generate(
            video_path=video_path,
            prompt="anime style, vibrant colors, detailed",
            strength=0.75,
            num_frames=20,  # 限制处理帧数以节省时间
            fps=8,
            output_name="comprehensive_video_to_video"
        )
    
    def demo_image_recognition(self, image_path: str):
        """演示图片识别"""
        print("\n" + "="*60)
        print("功能6: 图片识别")
        print("="*60)
        
        if not os.path.exists(image_path):
            print(f"⚠️  图片不存在: {image_path}")
            print("跳过此功能演示")
            return
        
        # 分类
        print("\n【图片分类】")
        classification_results = self.image_recognizer.classify(
            image_path,
            top_k=5
        )
        print("\n分类结果:")
        for label, confidence in classification_results:
            print(f"  {label}: {confidence:.2f}%")
        
        # 目标检测
        print("\n【目标检测】")
        detection_results = self.image_recognizer.detect(
            image_path,
            confidence_threshold=0.5,
            save_result=True
        )
        print(f"\n检测到 {len(detection_results)} 个物体:")
        for det in detection_results:
            print(f"  {det['label']}: {det['confidence']:.2f}%")
    
    def demo_video_recognition(self, video_path: str):
        """演示视频识别"""
        print("\n" + "="*60)
        print("功能7: 视频识别")
        print("="*60)
        
        if not os.path.exists(video_path):
            print(f"⚠️  视频不存在: {video_path}")
            print("跳过此功能演示")
            return
        
        # 分类
        print("\n【视频分类】")
        classification_results = self.video_recognizer.classify(
            video_path,
            sample_frames=10,
            top_k=5
        )
        print("\n分类结果:")
        for result in classification_results:
            print(f"  {result['label']}: {result['percentage']:.1f}% (置信度: {result['avg_confidence']:.2f}%)")
        
        # 目标检测
        print("\n【视频目标检测】")
        detection_results = self.video_recognizer.detect(
            video_path,
            sample_frames=10,
            confidence_threshold=0.5
        )
        print("\n检测结果:")
        for result in detection_results:
            print(f"  {result['label']}: {result['detections']} 次检测 (出现在 {result['percentage']:.1f}% 的帧中)")
    
    def demo_complete_workflow(self, image_path: str = None):
        """演示完整工作流"""
        print("\n" + "="*60)
        print("完整工作流演示")
        print("="*60)
        print("\n工作流：文本生成图片 -> 图片识别 -> 图片生成视频 -> 视频识别")
        
        # 步骤1: 文本生成图片
        print("\n【步骤1】文本生成图片")
        prompt = "a beautiful mountain landscape, lake, sunset, peaceful"
        image, image_path_generated = self.text_to_image_gen.generate(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            output_name="workflow_step1_image"
        )
        
        # 步骤2: 图片识别
        print("\n【步骤2】图片识别")
        if os.path.exists(image_path_generated):
            classification_results = self.image_recognizer.classify(
                image_path_generated,
                top_k=3
            )
            print("识别结果:")
            for label, confidence in classification_results:
                print(f"  {label}: {confidence:.2f}%")
        
        # 步骤3: 图片生成视频
        print("\n【步骤3】图片生成视频")
        if os.path.exists(image_path_generated):
            self.image_to_video_gen.generate(
                image_path=image_path_generated,
                num_frames=14,
                fps=7,
                output_name="workflow_step3_video"
            )
        
        print("\n✅ 完整工作流执行完成！")


def main():
    """主函数"""
    print("="*60)
    print("AI 综合功能演示")
    print("="*60)
    print("\n本示例将演示所有功能模块")
    print("包括：文生图、文生视频、图生图、图生视频、视频生视频、图片识别、视频识别")
    print("\n注意：")
    print("- 视频生成和处理需要较长时间")
    print("- 某些功能需要输入文件，如果文件不存在将跳过")
    print("- 首次运行需要下载模型，请确保网络连接稳定")
    
    generator = ComprehensiveAIGenerator()
    
    # 检查示例文件
    examples_images_dir = "examples/images"
    examples_videos_dir = "examples/videos"
    
    image_files = []
    video_files = []
    
    if os.path.exists(examples_images_dir):
        image_files = [f for f in os.listdir(examples_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if os.path.exists(examples_videos_dir):
        video_files = [f for f in os.listdir(examples_videos_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    # 演示各个功能
    print("\n" + "="*60)
    print("开始功能演示")
    print("="*60)
    
    # 1. 文本生成图片
    generator.demo_text_to_image()
    
    # 2. 图片生成图片（如果有输入图片）
    if image_files:
        image_path = os.path.join(examples_images_dir, image_files[0])
        generator.demo_image_to_image(image_path)
    
    # 3. 文本生成视频（注释掉以节省时间，取消注释以运行）
    # generator.demo_text_to_video()
    
    # 4. 图片生成视频（如果有输入图片）
    if image_files:
        image_path = os.path.join(examples_images_dir, image_files[0])
        # generator.demo_image_to_video(image_path)  # 注释掉以节省时间
    
    # 5. 视频生成视频（如果有输入视频）
    if video_files:
        video_path = os.path.join(examples_videos_dir, video_files[0])
        # generator.demo_video_to_video(video_path)  # 注释掉以节省时间
    
    # 6. 图片识别（如果有输入图片）
    if image_files:
        image_path = os.path.join(examples_images_dir, image_files[0])
        generator.demo_image_recognition(image_path)
    
    # 7. 视频识别（如果有输入视频）
    if video_files:
        video_path = os.path.join(examples_videos_dir, video_files[0])
        # generator.demo_video_recognition(video_path)  # 注释掉以节省时间
    
    # 完整工作流
    if image_files:
        image_path = os.path.join(examples_images_dir, image_files[0])
        # generator.demo_complete_workflow(image_path)  # 注释掉以节省时间
    
    print("\n" + "="*60)
    print("所有功能演示完成！")
    print("="*60)
    print("\n生成的图片保存在 outputs/images/ 目录下")
    print("生成的视频保存在 outputs/videos/ 目录下")
    print("\n提示：")
    print("- 取消注释相关代码可以运行视频生成和处理功能")
    print("- 视频功能需要较长时间，请耐心等待")
    print("- 确保有足够的磁盘空间和内存")


if __name__ == "__main__":
    main()
