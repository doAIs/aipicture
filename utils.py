"""
工具函数 - 辅助功能
"""
import os
from datetime import datetime
from PIL import Image
import torch


def save_image(image: Image.Image, filename: str = None, subfolder: str = None) -> str:
    """
    保存图片到输出目录
    
    Args:
        image: PIL Image对象
        filename: 文件名（不包含扩展名）
        subfolder: 子文件夹名称
    
    Returns:
        保存的文件路径
    """
    from config import OUTPUT_DIR
    
    # 创建子文件夹
    if subfolder:
        save_dir = os.path.join(OUTPUT_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = OUTPUT_DIR
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}"
    
    filepath = os.path.join(save_dir, f"{filename}.png")
    image.save(filepath)
    print(f"图片已保存到: {filepath}")
    return filepath


def load_image(image_path: str) -> Image.Image:
    """
    加载图片
    
    Args:
        image_path: 图片路径
    
    Returns:
        PIL Image对象
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    return image


def set_seed(seed: int):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")


def get_device() -> str:
    """
    获取可用设备（CPU或CUDA）
    
    Returns:
        设备名称
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("使用CPU（速度较慢，建议使用GPU）")
    return device

