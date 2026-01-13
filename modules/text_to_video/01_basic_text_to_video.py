"""
基础示例: 文本生成视频（最简单版本）
这是最基础的文本生成视频示例，适合初学者理解基本流程
"""

from diffusers import DiffusionPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import save_video, get_device, load_model_from_local_file, load_model_with_fallback
from config.modules_config import LOCAL_VIDEO_MODEL_PATH


def generate_video_from_text(prompt: str, output_name: str = None, num_frames: int = 16, fps: int = 8, local_model_path: str = None):
    """
    根据文本描述生成视频
    
    Args:
        prompt: 文本描述，例如 "a beautiful sunset over the ocean"
        output_name: 输出文件名（可选）
        num_frames: 视频帧数（默认16帧）
        fps: 帧率（默认8fps）
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
            # .frames 返回的是 List[List[Image]]，我们需要取第一个视频（下标 [0]）
            video_frames = output.frames[0]
        
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
    generate_video_from_text(prompt, "sunset_ocean", num_frames=16, fps=8)

