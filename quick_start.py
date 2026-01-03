"""
快速开始脚本
运行此脚本可以快速体验所有功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(__file__))

def print_banner():
    """打印欢迎横幅"""
    print("="*60)
    print("AI 多媒体生成与识别项目 - 快速开始")
    print("="*60)
    print("\n本脚本将引导你体验所有功能")
    print("请选择要运行的功能：\n")

def print_menu():
    """打印菜单"""
    print("=" * 60)
    print("生成功能：")
    print("  1. 基础文本生成图片")
    print("  2. 进阶文本生成图片（带参数控制）")
    print("  3. 基础图片生成图片（需要输入图片）")
    print("  4. 进阶图片生成图片（带参数控制）")
    print("  5. 文本生成视频（需要较长时间）")
    print("  6. 图片生成视频（需要输入图片和较长时间）")
    print("  7. 视频生成视频（需要输入视频和较长时间）")
    print("\n识别功能：")
    print("  8. 图片识别（分类和检测）")
    print("  9. 视频识别（分类和检测，需要输入视频）")
    print("\n其他：")
    print("  10. 综合示例（完整功能演示）")
    print("  0. 退出")
    print()

def run_basic_text_to_image():
    """运行基础文本生成图片"""
    print("\n" + "="*60)
    print("运行：基础文本生成图片")
    print("="*60)
    try:
        from modules.text_to_image import generate_image_from_text
        prompt = "a beautiful sunset over the ocean, peaceful, serene"
        generate_image_from_text(prompt, "quickstart_sunset")
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_advanced_text_to_image():
    """运行进阶文本生成图片"""
    print("\n" + "="*60)
    print("运行：进阶文本生成图片")
    print("="*60)
    try:
        from modules.text_to_image import AdvancedTextToImage
        generator = AdvancedTextToImage()
        generator.generate(
            prompt="a majestic lion standing on a rock, sunset, photorealistic",
            negative_prompt="blurry, low quality, distorted",
            output_name="quickstart_lion"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_basic_image_to_image():
    """运行基础图片生成图片"""
    print("\n" + "="*60)
    print("运行：基础图片生成图片")
    print("="*60)
    
    # 检查是否有输入图片
    example_dir = "examples/images"
    if not os.path.exists(example_dir):
        os.makedirs(example_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/images/ 目录，然后重新运行")
        print("或者先运行选项1生成一张图片作为输入")
        return
    
    try:
        from modules.image_to_image import generate_image_from_image
        image_path = os.path.join(example_dir, image_files[0])
        print(f"使用输入图片: {image_path}")
        generate_image_from_image(
            image_path=image_path,
            prompt="oil painting style, artistic, detailed",
            strength=0.7,
            output_name="quickstart_converted"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_advanced_image_to_image():
    """运行进阶图片生成图片"""
    print("\n" + "="*60)
    print("运行：进阶图片生成图片")
    print("="*60)
    
    example_dir = "examples/images"
    image_files = [f for f in os.listdir(example_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/images/ 目录，然后重新运行")
        return
    
    try:
        from modules.image_to_image import AdvancedImageToImage
        generator = AdvancedImageToImage()
        image_path = os.path.join(example_dir, image_files[0])
        print(f"使用输入图片: {image_path}")
        generator.generate(
            image_path=image_path,
            prompt="anime style, vibrant colors, detailed",
            strength=0.75,
            output_name="quickstart_anime"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_text_to_video():
    """运行文本生成视频"""
    print("\n" + "="*60)
    print("运行：文本生成视频")
    print("="*60)
    print("⚠️  注意：视频生成需要较长时间（可能需要几分钟到十几分钟）")
    confirm = input("是否继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        from modules.text_to_video import AdvancedTextToVideo
        generator = AdvancedTextToVideo()
        generator.generate(
            prompt="a beautiful sunset over the ocean, peaceful, serene",
            num_frames=16,
            fps=8,
            output_name="quickstart_text_to_video"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_image_to_video():
    """运行图片生成视频"""
    print("\n" + "="*60)
    print("运行：图片生成视频")
    print("="*60)
    
    example_dir = "examples/images"
    image_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/images/ 目录，然后重新运行")
        return
    
    print("⚠️  注意：视频生成需要较长时间（可能需要几分钟到十几分钟）")
    confirm = input("是否继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        from modules.image_to_video import AdvancedImageToVideo
        generator = AdvancedImageToVideo()
        image_path = os.path.join(example_dir, image_files[0])
        print(f"使用输入图片: {image_path}")
        generator.generate(
            image_path=image_path,
            num_frames=14,
            fps=7,
            output_name="quickstart_image_to_video"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_video_to_video():
    """运行视频生成视频"""
    print("\n" + "="*60)
    print("运行：视频生成视频")
    print("="*60)
    
    example_dir = "examples/videos"
    video_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(example_dir) else []
    
    if not video_files:
        print("\n⚠️  未找到输入视频！")
        print("请将视频放入 examples/videos/ 目录，然后重新运行")
        return
    
    print("⚠️  注意：视频处理需要较长时间（可能需要几十分钟）")
    confirm = input("是否继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        from modules.video_to_video import AdvancedVideoToVideo
        generator = AdvancedVideoToVideo()
        video_path = os.path.join(example_dir, video_files[0])
        print(f"使用输入视频: {video_path}")
        generator.generate(
            video_path=video_path,
            prompt="anime style, vibrant colors, detailed",
            strength=0.75,
            num_frames=20,  # 限制处理帧数以节省时间
            fps=8,
            output_name="quickstart_video_to_video"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_image_recognition():
    """运行图片识别"""
    print("\n" + "="*60)
    print("运行：图片识别")
    print("="*60)
    
    example_dir = "examples/images"
    image_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/images/ 目录，然后重新运行")
        return
    
    try:
        from modules.image_recognition import AdvancedImageRecognition
        recognizer = AdvancedImageRecognition()
        image_path = os.path.join(example_dir, image_files[0])
        print(f"使用输入图片: {image_path}")
        
        # 分类
        print("\n【图片分类】")
        results = recognizer.classify(image_path, top_k=5)
        print("\n分类结果:")
        for label, confidence in results:
            print(f"  {label}: {confidence:.2f}%")
        
        # 目标检测
        print("\n【目标检测】")
        detections = recognizer.detect(image_path, confidence_threshold=0.5, save_result=True)
        print(f"\n检测到 {len(detections)} 个物体:")
        for det in detections:
            print(f"  {det['label']}: {det['confidence']:.2f}%")
        
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_video_recognition():
    """运行视频识别"""
    print("\n" + "="*60)
    print("运行：视频识别")
    print("="*60)
    
    example_dir = "examples/videos"
    video_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(example_dir) else []
    
    if not video_files:
        print("\n⚠️  未找到输入视频！")
        print("请将视频放入 examples/videos/ 目录，然后重新运行")
        return
    
    try:
        from modules.video_recognition import AdvancedVideoRecognition
        recognizer = AdvancedVideoRecognition()
        video_path = os.path.join(example_dir, video_files[0])
        print(f"使用输入视频: {video_path}")
        
        # 分类
        print("\n【视频分类】")
        results = recognizer.classify(video_path, sample_frames=10, top_k=5)
        print("\n分类结果:")
        for result in results:
            print(f"  {result['label']}: {result['percentage']:.1f}% (置信度: {result['avg_confidence']:.2f}%)")
        
        # 目标检测
        print("\n【视频目标检测】")
        detections = recognizer.detect(video_path, sample_frames=10, confidence_threshold=0.5)
        print("\n检测结果:")
        for result in detections:
            print(f"  {result['label']}: {result['detections']} 次检测 (出现在 {result['percentage']:.1f}% 的帧中)")
        
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def run_comprehensive():
    """运行综合示例"""
    print("\n" + "="*60)
    print("运行：综合示例")
    print("="*60)
    print("这将演示所有功能模块")
    print("注意：视频生成和处理功能需要较长时间")
    confirm = input("是否继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        import importlib
        module = importlib.import_module("05_comprehensive_example")
        generator = module.ComprehensiveAIGenerator()
        
        # 运行基础功能演示
        generator.demo_text_to_image()
        
        # 如果有输入图片，演示图片相关功能
        example_images_dir = "examples/images"
        if os.path.exists(example_images_dir):
            image_files = [f for f in os.listdir(example_images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                image_path = os.path.join(example_images_dir, image_files[0])
                generator.demo_image_to_image(image_path)
                generator.demo_image_recognition(image_path)
        
        print("\n✅ 完成！")
        print("\n提示：取消注释综合示例中的代码可以运行视频相关功能")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print_banner()
    
    while True:
        print_menu()
        choice = input("请选择 (0-10): ").strip()
        
        if choice == "0":
            print("\n再见！")
            break
        elif choice == "1":
            run_basic_text_to_image()
        elif choice == "2":
            run_advanced_text_to_image()
        elif choice == "3":
            run_basic_image_to_image()
        elif choice == "4":
            run_advanced_image_to_image()
        elif choice == "5":
            run_text_to_video()
        elif choice == "6":
            run_image_to_video()
        elif choice == "7":
            run_video_to_video()
        elif choice == "8":
            run_image_recognition()
        elif choice == "9":
            run_video_recognition()
        elif choice == "10":
            run_comprehensive()
        else:
            print("\n❌ 无效选择，请重新输入")
        
        if choice != "0":
            input("\n按 Enter 返回菜单...")
            print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
        sys.exit(0)
