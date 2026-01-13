"""
项目验证脚本
检查项目文件是否完整，依赖是否安装
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[X] {description}: {filepath} (缺失)")
        return False

def check_directory_exists(dirpath, description):
    """检查目录是否存在"""
    if os.path.exists(dirpath):
        print(f"[OK] {description}: {dirpath}")
        return True
    else:
        print(f"[!] {description}: {dirpath} (将自动创建)")
        return False

def check_import(module_name, description):
    """检查模块是否可以导入"""
    try:
        __import__(module_name)
        print(f"[OK] {description}: {module_name}")
        return True
    except ImportError:
        print(f"[X] {description}: {module_name} (未安装)")
        return False

def main():
    """主验证函数"""
    print("="*60)
    print("项目验证")
    print("="*60)
    print()
    
    all_ok = True
    
    # 检查核心文件
    print("【核心文件】")
    print("-"*60)
    files_to_check = [
        ("README.md", "项目说明文档"),
        ("USAGE_GUIDE.md", "使用指南"),
        ("PROJECT_SUMMARY.md", "项目总结"),
        ("requirements.txt", "依赖文件"),
        ("modules_config.py", "配置文件"),
        ("modules_utils.py", "工具函数"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    print()
    
    # 检查示例文件
    print("【示例文件】")
    print("-"*60)
    example_files = [
        ("01_basic_text_to_image.py", "基础文本生成图片"),
        ("02_basic_image_to_image.py", "基础图片生成图片"),
        ("03_advanced_text_to_image.py", "进阶文本生成图片"),
        ("04_advanced_image_to_image.py", "进阶图片生成图片"),
        ("05_comprehensive_example.py", "综合示例"),
        ("quick_start.py", "快速开始脚本"),
    ]
    
    for filepath, desc in example_files:
        if not check_file_exists(filepath, desc):
            all_ok = False
    
    print()
    
    # 检查目录
    print("【目录结构】")
    print("-"*60)
    check_directory_exists("outputs", "输出目录")
    check_directory_exists("examples", "示例目录")
    check_file_exists("examples/README.md", "示例目录说明")
    
    print()
    
    # 检查依赖
    print("【Python 依赖】")
    print("-"*60)
    dependencies = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
    ]
    
    deps_ok = True
    for module, desc in dependencies:
        if not check_import(module, desc):
            deps_ok = False
    
    if not deps_ok:
        print()
        print("⚠️  部分依赖未安装，请运行: pip install -r requirements.txt")
        all_ok = False
    
    print()
    
    # 检查 CUDA
    print("【硬件支持】")
    print("-"*60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("[!] CUDA 不可用，将使用 CPU（速度较慢）")
    except:
        print("[!] 无法检查 CUDA 状态")
    
    print()
    
    # 总结
    print("="*60)
    if all_ok and deps_ok:
        print("[OK] 项目验证通过！所有文件完整，依赖已安装。")
        print()
        print("可以开始使用项目：")
        print("  1. 运行快速开始: python quick_start.py")
        print("  2. 运行基础示例: python 01_basic_text_to_image.py")
        return 0
    else:
        print("[X] 项目验证未完全通过，请检查上述问题。")
        print()
        if not deps_ok:
            print("建议操作：")
            print("  1. 安装依赖: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

