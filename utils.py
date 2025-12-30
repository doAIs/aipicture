"""
工具函数 - 辅助功能
"""
import os
from datetime import datetime
from PIL import Image
import torch
import numpy as np
from typing import List, Tuple, Optional


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
    from config import OUTPUT_IMAGES_DIR
    
    # 创建子文件夹
    if subfolder:
        save_dir = os.path.join(OUTPUT_IMAGES_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = OUTPUT_IMAGES_DIR
        os.makedirs(save_dir, exist_ok=True)
    
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


def save_video(frames: List[Image.Image], filename: str = None, subfolder: str = None, fps: int = 8) -> str:
    """
    保存视频帧序列为视频文件
    
    Args:
        frames: PIL Image对象列表
        filename: 文件名（不包含扩展名）
        subfolder: 子文件夹名称
        fps: 帧率
    
    Returns:
        保存的文件路径
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("请安装 imageio 和 imageio-ffmpeg: pip install imageio imageio-ffmpeg")
    
    from config import OUTPUT_VIDEOS_DIR
    
    # 创建子文件夹
    if subfolder:
        save_dir = os.path.join(OUTPUT_VIDEOS_DIR, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = OUTPUT_VIDEOS_DIR
        os.makedirs(save_dir, exist_ok=True)
    
    # 生成文件名
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}"
    
    filepath = os.path.join(save_dir, f"{filename}.mp4")
    
    # 转换为numpy数组
    frame_arrays = [np.array(frame) for frame in frames]
    
    # 保存为视频
    imageio.mimwrite(filepath, frame_arrays, fps=fps, codec='libx264', quality=8)
    print(f"视频已保存到: {filepath}")
    return filepath


def load_video(video_path: str) -> List[Image.Image]:
    """
    加载视频文件，返回帧列表
    
    Args:
        video_path: 视频路径
    
    Returns:
        PIL Image对象列表
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("请安装 imageio 和 imageio-ffmpeg: pip install imageio imageio-ffmpeg")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 读取视频帧
    reader = imageio.get_reader(video_path)
    frames = []
    for frame in reader:
        frames.append(Image.fromarray(frame))
    
    reader.close()
    print(f"已加载视频: {video_path}，共 {len(frames)} 帧")
    return frames


def video_to_frames(video_path: str, output_dir: str = None, max_frames: int = None) -> List[Image.Image]:
    """
    将视频转换为帧序列
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录（可选，如果提供则保存帧）
        max_frames: 最大帧数（可选）
    
    Returns:
        PIL Image对象列表
    """
    frames = load_video(video_path)
    
    if max_frames and len(frames) > max_frames:
        # 均匀采样
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
    
    return frames


def frames_to_video(frames: List[Image.Image], output_path: str, fps: int = 8) -> str:
    """
    将帧序列转换为视频
    
    Args:
        frames: PIL Image对象列表
        output_path: 输出视频路径
        fps: 帧率
    
    Returns:
        输出视频路径
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("请安装 imageio 和 imageio-ffmpeg: pip install imageio imageio-ffmpeg")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    frame_arrays = [np.array(frame) for frame in frames]
    imageio.mimwrite(output_path, frame_arrays, fps=fps, codec='libx264', quality=8)
    print(f"视频已保存到: {output_path}")
    return output_path


def resize_image(image: Image.Image, size: Tuple[int, int], keep_aspect: bool = True) -> Image.Image:
    """
    调整图片大小
    
    Args:
        image: PIL Image对象
        size: 目标尺寸 (width, height)
        keep_aspect: 是否保持宽高比
    
    Returns:
        调整后的PIL Image对象
    """
    if keep_aspect:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        # 创建新图片，居中放置
        new_image = Image.new("RGB", size, (0, 0, 0))
        paste_x = (size[0] - image.size[0]) // 2
        paste_y = (size[1] - image.size[1]) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def set_seed(seed: int):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    import random
    
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


def get_image_info(image_path: str) -> dict:
    """
    获取图片信息
    
    Args:
        image_path: 图片路径
    
    Returns:
        包含图片信息的字典
    """
    image = load_image(image_path)
    return {
        "path": image_path,
        "size": image.size,
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "format": image.format
    }


def get_video_info(video_path: str) -> dict:
    """
    获取视频信息
    
    Args:
        video_path: 视频路径
    
    Returns:
        包含视频信息的字典
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("请安装 imageio 和 imageio-ffmpeg: pip install imageio imageio-ffmpeg")
    
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    frames = len(list(reader))
    reader.close()
    
    return {
        "path": video_path,
        "fps": meta.get("fps", 0),
        "duration": meta.get("duration", 0),
        "size": meta.get("size", (0, 0)),
        "frames": frames
    }

