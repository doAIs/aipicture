"""
基础示例: 文本生成视频（最简单版本）
这是最基础的文本生成视频示例，适合初学者理解基本流程
"""

from diffusers import DiffusionPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import save_video, get_device


def generate_video_from_text(prompt: str, output_name: str = None, num_frames: int = 16, fps: int = 8):
    """
    根据文本描述生成视频
    
    Args:
        prompt: 文本描述，例如 "a beautiful sunset over the ocean"
        output_name: 输出文件名（可选）
        num_frames: 视频帧数（默认16帧）
        fps: 帧率（默认8fps）
    """
    print(f"\n开始生成视频...")
    print(f"提示词: {prompt}")
    print(f"帧数: {num_frames}, 帧率: {fps}")
    
    # 获取设备
    device = get_device()
    
    # 加载模型（首次运行会自动下载，需要一些时间）
    print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
    print("注意：视频生成模型较大，下载可能需要较长时间")
    
    # 根据设备选择数据类型
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # 使用文本生成视频模型
    # 注意：这里使用一个较小的模型作为示例
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch_dtype,
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
            video_frames = pipe(
                prompt,
                num_inference_steps=50,
                num_frames=num_frames
            ).frames
        
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

