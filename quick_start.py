"""
快速开始脚本
运行此脚本可以快速体验所有功能
"""

import os
import sys

def print_banner():
    """打印欢迎横幅"""
    print("="*60)
    print("AI 图片生成项目 - 快速开始")
    print("="*60)
    print("\n本脚本将引导你体验所有功能")
    print("请选择要运行的功能：\n")

def print_menu():
    """打印菜单"""
    print("1. 基础文本生成图片（最简单）")
    print("2. 基础图片生成图片（需要输入图片）")
    print("3. 进阶文本生成图片（带参数控制）")
    print("4. 进阶图片生成图片（带参数控制）")
    print("5. 综合示例（完整工作流）")
    print("6. 运行所有示例")
    print("0. 退出")
    print()

def run_basic_text_to_image():
    """运行基础文本生成图片"""
    print("\n" + "="*60)
    print("运行：基础文本生成图片")
    print("="*60)
    try:
        from 01_basic_text_to_image import generate_image_from_text
        prompt = "a beautiful sunset over the ocean, peaceful, serene"
        generate_image_from_text(prompt, "quickstart_sunset")
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def run_basic_image_to_image():
    """运行基础图片生成图片"""
    print("\n" + "="*60)
    print("运行：基础图片生成图片")
    print("="*60)
    
    # 检查是否有输入图片
    example_dir = "examples"
    if not os.path.exists(example_dir):
        os.makedirs(example_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(example_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/ 目录，然后重新运行")
        print("或者先运行选项1生成一张图片作为输入")
        return
    
    try:
        from 02_basic_image_to_image import generate_image_from_image
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

def run_advanced_text_to_image():
    """运行进阶文本生成图片"""
    print("\n" + "="*60)
    print("运行：进阶文本生成图片")
    print("="*60)
    try:
        from 03_advanced_text_to_image import AdvancedTextToImage
        generator = AdvancedTextToImage()
        generator.generate(
            prompt="a majestic lion standing on a rock, sunset, photorealistic",
            negative_prompt="blurry, low quality, distorted",
            output_name="quickstart_lion"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def run_advanced_image_to_image():
    """运行进阶图片生成图片"""
    print("\n" + "="*60)
    print("运行：进阶图片生成图片")
    print("="*60)
    
    example_dir = "examples"
    image_files = [f for f in os.listdir(example_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if not image_files:
        print("\n⚠️  未找到输入图片！")
        print("请将图片放入 examples/ 目录，然后重新运行")
        return
    
    try:
        from 04_advanced_image_to_image import AdvancedImageToImage
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

def run_comprehensive():
    """运行综合示例"""
    print("\n" + "="*60)
    print("运行：综合示例")
    print("="*60)
    try:
        from 05_comprehensive_example import ComprehensiveImageGenerator
        generator = ComprehensiveImageGenerator()
        
        # 运行一个简单示例
        generator.text_to_image(
            prompt="a cute robot, cartoon style, colorful, detailed",
            negative_prompt="blurry, low quality",
            output_name="quickstart_robot"
        )
        print("\n✅ 完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")

def run_all():
    """运行所有示例"""
    print("\n" + "="*60)
    print("运行所有示例")
    print("="*60)
    print("\n这将依次运行所有功能，可能需要较长时间...")
    input("按 Enter 继续...")
    
    run_basic_text_to_image()
    input("\n按 Enter 继续下一个示例...")
    
    run_advanced_text_to_image()
    input("\n按 Enter 继续下一个示例...")
    
    # 如果有输入图片，运行图片生成图片示例
    example_dir = "examples"
    image_files = [f for f in os.listdir(example_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(example_dir) else []
    
    if image_files:
        run_basic_image_to_image()
        input("\n按 Enter 继续下一个示例...")
        run_advanced_image_to_image()
    
    run_comprehensive()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)

def main():
    """主函数"""
    print_banner()
    
    while True:
        print_menu()
        choice = input("请选择 (0-6): ").strip()
        
        if choice == "0":
            print("\n再见！")
            break
        elif choice == "1":
            run_basic_text_to_image()
        elif choice == "2":
            run_basic_image_to_image()
        elif choice == "3":
            run_advanced_text_to_image()
        elif choice == "4":
            run_advanced_image_to_image()
        elif choice == "5":
            run_comprehensive()
        elif choice == "6":
            run_all()
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

