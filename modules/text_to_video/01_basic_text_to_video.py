"""
基础示例: 文本生成视频（最简单版本）
这是最基础的文本生成视频示例，适合初学者理解基本流程
"""
from datetime import datetime

from diffusers import DiffusionPipeline
import torch
import sys
import os

from diffusers.utils import export_to_video

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import save_video, get_device, load_model_from_local_file, load_model_with_fallback
from config.modules_config import LOCAL_VIDEO_MODEL_PATH, OUTPUT_VIDEOS_DIR


def generate_video_from_text(prompt: str, output_name: str = None, num_frames: int = 16, fps: int = 8, local_model_path: str = None):
    """
    视频时长(秒) = num_frames ÷ fps
    根据文本描述生成视频
    
    Args:
        prompt: 文本描述，例如 "a beautiful sunset over the ocean"
        output_name: 输出文件名（可选）
        num_frames: 视频帧数（默认16帧）
        作用：生成的视频包含多少帧图像
        默认值：16 帧
         影响：🎬 决定视频素材量：帧数越多，视频内容越丰富
         ⏱️ 影响生成时间：帧数越多，生成越慢
         💾 影响内存占用：帧数越多，需要更多显存/内存
         📏 与时长相关（见下方公式）

        fps: 帧率（默认8fps）
        作用：Frames Per Second（每秒播放多少帧）
             默认值：8 fps
             影响：🎞️ 决定播放速度：fps 越高，视频越流畅
             📏 与时长相关（见下方公式）
             🎥 常见标准：8 fps：较慢，AI视频常用
                        24 fps：电影标准
                        30 fps：视频标准
                        60 fps：高清流畅
        local_model_path: 本地模型路径（可选）
                         - 如果为 None，则从配置文件 config.LOCAL_VIDEO_MODEL_PATH 读取
                         - 如果为 "" 或空字符串，则禁用本地模型，仅使用在线模型
                         - 如果指定路径，则使用指定的路径
    """
    print(f"\n开始生成视频...")
    print(f"提示词: {prompt}")
    print(f"帧数: {num_frames}, 帧率: {fps}")
    
    # 获取设备
    device = get_device()
    
    # 确定本地模型路径的优先级
    if local_model_path is not None:
        model_path = local_model_path if local_model_path else None
    else:
        model_path = LOCAL_VIDEO_MODEL_PATH if LOCAL_VIDEO_MODEL_PATH else None
    
    # 加载模型（优先使用本地模型，如果不存在则使用在线模型）
    if model_path:
        print(f"\n本地模型路径: {model_path}")
    else:
        print("\n本地模型: 已禁用（仅使用在线模型）")
    
    # 根据设备选择数据类型
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # 准备模型加载参数
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    
    # 使用文本生成视频模型
    # 优先尝试加载本地模型，如果不存在则使用在线模型
    try:
        # 优先尝试加载本地模型
        if model_path and os.path.exists(model_path):
            print(f"\n✅ 检测到本地模型路径: {model_path}")
            print("   优先使用本地离线模型...")
            try:
                pipe = load_model_from_local_file(
                    DiffusionPipeline,
                    model_path,
                    **model_kwargs
                )
                pipe = pipe.to(device)
                print("✅ 本地模型加载成功！")
            except Exception as e:
                print(f"⚠️  本地模型加载失败: {e}")
                print(f"   回退到在线模型: ali-vilab/text-to-video-ms-1.7b")
                # 本地模型加载失败，回退到在线模型
                pipe = load_model_with_fallback(
                    DiffusionPipeline,
                    "ali-vilab/text-to-video-ms-1.7b",
                    **model_kwargs
                )
                pipe = pipe.to(device)
        elif model_path:
            # 本地模型路径已配置但不存在，直接使用在线模型
            print(f"\n⚠️  本地模型路径不存在: {model_path}")
            print(f"   使用在线模型: ali-vilab/text-to-video-ms-1.7b")
            pipe = load_model_with_fallback(
                DiffusionPipeline,
                "ali-vilab/text-to-video-ms-1.7b",
                **model_kwargs
            )
            pipe = pipe.to(device)
        else:
            # 本地模型未配置，直接使用在线模型
            print(f"\n📡 使用在线模型: ali-vilab/text-to-video-ms-1.7b")
            print("   注意：视频生成模型较大，下载可能需要较长时间")
            pipe = load_model_with_fallback(
                DiffusionPipeline,
                "ali-vilab/text-to-video-ms-1.7b",
                **model_kwargs
            )
            pipe = pipe.to(device)
        
        # 优化：启用内存高效注意力
        try:
            pipe.enable_attention_slicing()
            print("已启用注意力切片（节省内存）")
        except:
            pass
        
        print("模型加载完成！")
        
        # 生成视频
        print(f"\n正在生成视频（这可能需要几分钟）...")
        print("视频生成比图片生成慢得多，请耐心等待...")

        with torch.no_grad():
            output = pipe(
                prompt,
                num_inference_steps=50,
                num_frames=num_frames
            )

            ##使用官方API
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}"
            save_dir = os.path.join(OUTPUT_VIDEOS_DIR, "basic_text_to_video")
            filepath = os.path.join(save_dir, f"{filename}.mp4")
            os.makedirs(save_dir, exist_ok=True)
            video_path = export_to_video(video_frames=output.frames[0],output_video_path=filepath ,quality=5,fps=fps)
            print(f"================视频保存路径: {video_path}")
            
            # 安全获取视频帧
            # output.frames 的可能结构：
            # 1. numpy.ndarray: shape=(1, num_frames, H, W, 3) - 最常见
            # 2. List[List[Image]]: 批次列表，每个批次包含一个视频的帧序列
            # 3. List[Image]: 直接的帧列表
            
            import numpy as np
            
            if hasattr(output, 'frames'):
                frames = output.frames

                
                # 情况1: numpy数组格式 (shape: [batch, num_frames, H, W, C])
                if isinstance(frames, np.ndarray):
                    print(f"检测到numpy数组格式，shape: {frames.shape}")
                    # 通常是 (1, num_frames, height, width, 3)
                    if len(frames.shape) == 5:
                        # 取第一个批次: [num_frames, H, W, C]
                        frames_batch = frames[0]
                        # 转换为PIL Image列表
                        from PIL import Image
                        video_frames = []
                        for i in range(frames_batch.shape[0]):
                            frame_data = frames_batch[i]  # [H, W, C]
                            # 确保数据在0-1范围内，然后转换为0-255
                            if frame_data.max() <= 1.0:
                                frame_data = (frame_data * 255).astype(np.uint8)
                            else:
                                frame_data = frame_data.astype(np.uint8)
                            # 转换为PIL Image
                            img = Image.fromarray(frame_data)
                            video_frames.append(img)
                    else:
                        raise ValueError(f"意外的numpy数组形状: {frames.shape}，期望5维数组 [batch, frames, H, W, C]")
                
                # 情况2: 列表格式
                elif isinstance(frames, list) and len(frames) > 0:
                    # 检查是否为嵌套列表（批次结构）
                    if isinstance(frames[0], list):
                        video_frames = frames[0]  # 取第一个批次
                    else:
                        video_frames = frames  # 直接就是帧列表
                
                else:
                    raise ValueError(f"模型输出的 frames 格式不支持，类型: {type(frames)}")
            
            elif hasattr(output, 'images'):
                # 某些模型可能使用 images 属性
                video_frames = output.images
            
            else:
                raise ValueError("无法从模型输出中获取视频帧，输出类型: " + str(type(output)))
        
        # 调试信息：检查帧数据
        print(f"\n生成了 {len(video_frames)} 帧")
        if len(video_frames) > 0:
            first_frame = video_frames[0]
            print(f"帧类型: {type(first_frame)}")
            if hasattr(first_frame, 'size'):
                print(f"帧尺寸: {first_frame.size}")
            if hasattr(first_frame, 'mode'):
                print(f"帧模式: {first_frame.mode}")
            # 转换为numpy检查数据范围
            import numpy as np
            frame_array = np.array(first_frame)
            print(f"数据类型: {frame_array.dtype}")
            print(f"数据范围: [{frame_array.min():.4f}, {frame_array.max():.4f}]")
            print(f"数据形状: {frame_array.shape}")
        
        # 保存视频
        filepath = save_video(video_frames, output_name, "basic_text_to_video", fps=fps)
        print(f"\n✅ 生成完成！")
        return video_frames, filepath
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示：")
        print("1. 确保已安装所有依赖：pip install -r requirements.txt")
        print("2. 视频生成需要大量内存，建议使用GPU")
        print("3. 如果内存不足，可以尝试减少num_frames参数")
        raise


if __name__ == "__main__":
    # 示例: 生成一个简单的视频
    prompt = "a beautiful sunset over the ocean, peaceful, serene"
    generate_video_from_text(prompt, "sunset_ocean2", num_frames=16, fps=8)

