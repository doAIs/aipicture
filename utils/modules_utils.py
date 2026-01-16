"""
工具函数 - 辅助功能
"""
import os
from datetime import datetime
from PIL import Image
import torch
import numpy as np
from typing import List, Tuple, Optional
from config.modules_config import OUTPUT_IMAGES_DIR


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
    
    from config.modules_config import OUTPUT_VIDEOS_DIR
    
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

    print(f"frames data: {frames}")
    
    # 转换为numpy数组，确保数据格式正确
    frame_arrays = []
    for frame in frames:
        # 确保是PIL Image对象
        print(f"frame data: {frames}")
        if isinstance(frame, Image.Image):
            # 转换为RGB模式
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            # 转换为numpy数组，确保数据类型为uint8，范围在0-255
            frame_array = np.array(frame)
        else:
            # 如果已经是numpy数组
            frame_array = np.array(frame)
        
        # 确保数据类型正确
        if frame_array.dtype != np.uint8:
            # 如果是浮点数类型，需要转换为uint8
            if np.issubdtype(frame_array.dtype, np.floating):
                # 浮点数类型，检查范围
                if frame_array.min() >= 0.0 and frame_array.max() <= 1.0 + 1e-5:  # 允许小的浮点误差
                    # 0-1范围，转换为0-255
                    frame_array = (np.clip(frame_array, 0, 1) * 255).astype(np.uint8)
                else:
                    # 其他范围，直接裁剪到0-255
                    frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            else:
                # 整数类型，直接转换
                frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        
        frame_arrays.append(frame_array)
    
    # 保存为视频，使用更高质量的设置
    # quality参数：对于H.264，值越小质量越高（0最好），推荐使用5-10
    # 添加像素格式参数确保兼容性
    imageio.mimwrite(
        filepath, 
        frame_arrays, 
        fps=fps, 
        codec='libx264',
        quality=5,  # 提高视频质量（0-10，值越小质量越高）
        pixelformat='yuv420p',  # 添加像素格式，提高兼容性
        macro_block_size=1  # 避免尺寸问题
    )
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
    
    # 转换为numpy数组，确保数据格式正确
    frame_arrays = []
    for frame in frames:
        # 确保是PIL Image对象
        if isinstance(frame, Image.Image):
            # 转换为RGB模式
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            # 转换为numpy数组
            frame_array = np.array(frame)
        else:
            # 如果已经是numpy数组
            frame_array = np.array(frame)
        
        # 确保数据类型正确
        if frame_array.dtype != np.uint8:
            # 如果是浮点数类型，需要转换为uint8
            if np.issubdtype(frame_array.dtype, np.floating):
                # 浮点数类型，检查范围
                if frame_array.min() >= 0.0 and frame_array.max() <= 1.0 + 1e-5:  # 允许小的浮点误差
                    # 0-1范围，转换为0-255
                    frame_array = (np.clip(frame_array, 0, 1) * 255).astype(np.uint8)
                else:
                    # 其他范围，直接裁剪到0-255
                    frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            else:
                # 整数类型，直接转换
                frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
        
        frame_arrays.append(frame_array)
    
    # 保存为视频，使用高质量设置
    imageio.mimwrite(
        output_path, 
        frame_arrays, 
        fps=fps, 
        codec='libx264', 
        quality=5,
        pixelformat='yuv420p',
        macro_block_size=1
    )
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


def load_model_from_local_file(pipeline_class, model_path: str, **kwargs):
    """
    从本地文件加载模型（支持 .safetensors 或 .ckpt 文件）
    
    Args:
        pipeline_class: Pipeline类（如 StableDiffusionPipeline）
        model_path: 本地模型文件路径（.safetensors 或 .ckpt）或模型目录路径
        **kwargs: 传递给from_pretrained或from_single_file的其他参数
    
    Returns:
        加载的pipeline对象
    
    Raises:
        FileNotFoundError: 如果模型文件不存在
        ValueError: 如果文件格式不支持
    """
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"\n正在从本地文件加载模型: {model_path}")
    
    # 检查是否是单个文件（.safetensors 或 .ckpt）
    if os.path.isfile(model_path):
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext in ['.safetensors', '.ckpt']:
            # 使用 from_single_file 方法（diffusers 0.21.0+ 支持）
            try:
                print(f"检测到 {file_ext} 格式，使用 from_single_file 加载...")
                pipe = pipeline_class.from_single_file(model_path, **kwargs)
                print("✅ 从本地文件加载模型成功！")
                return pipe
            except AttributeError:
                # 如果 diffusers 版本不支持 from_single_file，尝试其他方法
                print("⚠️  当前 diffusers 版本不支持 from_single_file，尝试其他方法...")
                raise ValueError(
                    f"当前 diffusers 版本不支持从 {file_ext} 文件加载。"
                    f"请升级 diffusers 到 0.21.0 或更高版本，或使用模型目录格式。"
                )
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持格式: .safetensors, .ckpt")
    
    # 如果是目录，使用 from_pretrained
    elif os.path.isdir(model_path):
        print("检测到模型目录，使用 from_pretrained 加载...")
        try:
            pipe = pipeline_class.from_pretrained(model_path, **kwargs)
            print("✅ 从本地目录加载模型成功！")
            return pipe
        except Exception as e:
            raise OSError(f"从本地目录加载模型失败: {e}")
    
    else:
        raise ValueError(f"无效的模型路径: {model_path}")


def load_model_with_fallback(pipeline_class, model_name: str, **kwargs):
    """
    加载模型，带错误处理和离线模式支持
    
    Args:
        pipeline_class: Pipeline类（如 StableDiffusionPipeline）
        model_name: 模型名称（HuggingFace ID）或本地路径
        **kwargs: 传递给from_pretrained的其他参数
    
    Returns:
        加载的pipeline对象
    
    Raises:
        OSError: 如果模型无法加载且没有本地缓存
    """
    import os
    
    # 检查是否是本地文件路径
    if os.path.exists(model_name):
        return load_model_from_local_file(pipeline_class, model_name, **kwargs)
    
    print(f"\n正在加载模型: {model_name}")
    print("（首次运行需要下载，请耐心等待）...")
    
    # 检查是否启用离线模式
    offline_mode = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    local_files_only = os.getenv("HF_HUB_LOCAL_FILES_ONLY", "0") == "1"
    
    if offline_mode or local_files_only:
        print("⚠️  离线模式：仅使用本地缓存的模型")
        kwargs["local_files_only"] = True
    
    try:
        # 尝试加载模型
        try:
            pipe = pipeline_class.from_pretrained(model_name, **kwargs)
            print("✅ 模型加载成功！")
            return pipe
        except (OSError, Exception) as e:
            # 检查是否是网络相关错误
            error_str = str(e).lower()
            is_network_error = any(keyword in error_str for keyword in [
                'timeout', 'connection', 'connect', 'network', 
                'max retries', 'connection pool', 'huggingface.co'
            ])
            
            if not is_network_error:
                # 非网络错误，直接抛出
                raise
            
            # 网络错误，尝试使用本地缓存
            print("\n⚠️  网络连接失败，尝试使用本地缓存的模型...")
            print(f"   错误信息: {str(e)[:100]}...")
            
            # 检查本地缓存
            from huggingface_hub import HfFolder
            cache_dir = HfFolder.get_cache_dir()
            model_cache_path = os.path.join(cache_dir, "models--" + model_name.replace("/", "--"))
            
            if os.path.exists(model_cache_path):
                print(f"   找到本地缓存: {model_cache_path}")
                try:
                    kwargs["local_files_only"] = True
                    pipe = pipeline_class.from_pretrained(model_name, **kwargs)
                    print("✅ 从本地缓存加载模型成功！")
                    return pipe
                except Exception as cache_error:
                    print(f"   ❌ 本地缓存加载失败: {cache_error}")
            
            # 如果都失败了，提供详细的错误信息
            print("\n" + "="*60)
            print("❌ 模型加载失败")
            print("="*60)
            print("\n可能的原因：")
            print("1. 网络连接问题（无法连接到 huggingface.co）")
            print("2. 模型未下载且本地无缓存")
            print("\n解决方案：")
            print("1. 检查网络连接，确保可以访问 huggingface.co")
            print("2. 使用 VPN 或代理（如果在受限网络环境中）")
            print("3. 手动下载模型到本地缓存目录")
            print(f"   缓存目录: {cache_dir}")
            print("4. 设置环境变量启用离线模式（如果已有本地缓存）:")
            print("   set HF_HUB_LOCAL_FILES_ONLY=1")
            print("5. 使用本地模型文件路径（支持 .safetensors 和 .ckpt 格式）")
            print("="*60)
            raise OSError(
                f"无法加载模型 '{model_name}': 网络连接失败且本地无缓存。"
                f"请检查网络连接或手动下载模型。"
            ) from e
            
    except Exception as e:
        # 其他错误（非网络错误）
        if "网络" not in str(e) and "connection" not in str(e).lower():
            print(f"\n❌ 模型加载失败: {e}")
        raise


def load_transformers_model_with_fallback(
    processor_class,
    model_class,
    model_name: str,
    local_model_path: str = None,
    **kwargs
):
    """
    加载 Transformers 模型（如 ViT），优先使用本地模型
    
    Args:
        processor_class: 处理器类（如 AutoImageProcessor）
        model_class: 模型类（如 AutoModelForImageClassification）
        model_name: 在线模型名称（HuggingFace ID）
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则不使用本地模型
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则优先使用本地模型
        **kwargs: 传递给 from_pretrained 的其他参数
    
    Returns:
        (processor, model) 元组
    """
    import os
    
    processor = None
    model = None
    loaded_from_local = False
    
    # 确定本地模型路径
    effective_local_path = local_model_path if local_model_path else None
    
    # 优先尝试加载本地模型
    if effective_local_path and os.path.exists(effective_local_path):
        print(f"\n✅ 检测到本地模型路径: {effective_local_path}")
        print("   优先使用本地离线模型...")
        try:
            processor = processor_class.from_pretrained(effective_local_path, **kwargs)
            model = model_class.from_pretrained(effective_local_path, **kwargs)
            print("✅ 本地模型加载成功！")
            loaded_from_local = True
        except Exception as e:
            print(f"⚠️  本地模型加载失败: {e}")
            print(f"   回退到在线模型: {model_name}")
    elif effective_local_path:
        print(f"\n⚠️  本地模型路径不存在: {effective_local_path}")
        print(f"   使用在线模型: {model_name}")
    else:
        print(f"\n📡 使用在线模型: {model_name}")
    
    # 如果本地模型加载失败或未配置，使用在线模型
    if not loaded_from_local:
        print("（首次运行需要下载，请耐心等待）...")
        
        # 检查是否启用离线模式
        offline_mode = os.getenv("HF_HUB_OFFLINE", "0") == "1"
        local_files_only = os.getenv("HF_HUB_LOCAL_FILES_ONLY", "0") == "1"
        
        if offline_mode or local_files_only:
            print("⚠️  离线模式：仅使用本地缓存的模型")
            kwargs["local_files_only"] = True
        
        try:
            processor = processor_class.from_pretrained(model_name, **kwargs)
            model = model_class.from_pretrained(model_name, **kwargs)
            print("✅ 模型加载成功！")
        except Exception as e:
            error_str = str(e).lower()
            is_network_error = any(keyword in error_str for keyword in [
                'timeout', 'connection', 'connect', 'network', 
                'max retries', 'connection pool', 'huggingface.co'
            ])
            
            if is_network_error:
                # 尝试使用本地缓存
                print("\n⚠️  网络连接失败，尝试使用本地缓存的模型...")
                try:
                    kwargs["local_files_only"] = True
                    processor = processor_class.from_pretrained(model_name, **kwargs)
                    model = model_class.from_pretrained(model_name, **kwargs)
                    print("✅ 从本地缓存加载模型成功！")
                except Exception as cache_error:
                    print(f"❌ 本地缓存加载失败: {cache_error}")
                    raise OSError(
                        f"无法加载模型 '{model_name}': 网络连接失败且本地无缓存。"
                        f"请检查网络连接或配置本地模型路径。"
                    ) from e
            else:
                raise
    
    return processor, model


def load_yolo_model_with_fallback(
    model_name: str = "yolov8n.pt",
    local_model_path: str = None
):
    """
    加载 YOLO 模型，优先使用本地模型
    
    Args:
        model_name: 在线模型名称（默认 yolov8n.pt）
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则不使用本地模型
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则优先使用本地模型
    
    Returns:
        YOLO 模型对象
    """
    import os
    
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("请安装 ultralytics: pip install ultralytics")
    
    model = None
    effective_local_path = local_model_path if local_model_path else None
    
    # 优先尝试加载本地模型
    if effective_local_path and os.path.exists(effective_local_path):
        print(f"\n✅ 检测到本地YOLO模型路径: {effective_local_path}")
        print("   优先使用本地离线模型...")
        try:
            model = YOLO(effective_local_path)
            print("✅ 本地YOLO模型加载成功！")
            return model
        except Exception as e:
            print(f"⚠️  本地YOLO模型加载失败: {e}")
            print(f"   回退到在线模型: {model_name}")
    elif effective_local_path:
        print(f"\n⚠️  本地YOLO模型路径不存在: {effective_local_path}")
        print(f"   使用在线模型: {model_name}")
    else:
        print(f"\n📡 使用在线模型: {model_name}")
    
    # 使用在线模型
    print("（首次运行需要下载，请耐心等待）...")
    try:
        model = YOLO(model_name)
        print("✅ YOLO模型加载成功！")
        return model
    except Exception as e:
        print(f"\n❌ YOLO模型加载失败: {e}")
        print("\n提示：")
        print("1. 检查网络连接")
        print("2. 配置本地模型路径以避免网络下载")
        raise


def load_diffusion_pipeline_with_fallback(
    pipeline_class,
    model_name: str,
    local_model_path: str = None,
    **kwargs
):
    """
    加载 Diffusers Pipeline，优先使用本地模型
    
    Args:
        pipeline_class: Pipeline类（如 StableDiffusionImg2ImgPipeline）
        model_name: 在线模型名称（HuggingFace ID）
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则不使用本地模型
                         - 如果为 "" 或空字符串，则禁用本地模型
                         - 如果指定路径，则优先使用本地模型
        **kwargs: 传递给 from_pretrained 的其他参数
    
    Returns:
        加载的 pipeline 对象
    """
    import os
    
    pipe = None
    loaded_from_local = False
    effective_local_path = local_model_path if local_model_path else None
    
    # 优先尝试加载本地模型
    if effective_local_path and os.path.exists(effective_local_path):
        print(f"\n✅ 检测到本地模型路径: {effective_local_path}")
        print("   优先使用本地离线模型...")
        try:
            pipe = load_model_from_local_file(pipeline_class, effective_local_path, **kwargs)
            print("✅ 本地模型加载成功！")
            loaded_from_local = True
        except Exception as e:
            print(f"⚠️  本地模型加载失败: {e}")
            print(f"   回退到在线模型: {model_name}")
    elif effective_local_path:
        print(f"\n⚠️  本地模型路径不存在: {effective_local_path}")
        print(f"   使用在线模型: {model_name}")
    else:
        print(f"\n📡 使用在线模型: {model_name}")
        print("   注意：模型可能较大，下载可能需要较长时间")
    
    # 如果本地模型加载失败或未配置，使用在线模型
    if not loaded_from_local:
        pipe = load_model_with_fallback(pipeline_class, model_name, **kwargs)
    
    return pipe