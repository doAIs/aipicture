"""
基础示例: 视频生成视频（最简单版本）
根据输入视频和文本提示词生成新视频
"""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import load_video, save_video, get_device, resize_image


def generate_video_from_video(
    video_path: str,
    prompt: str,
    output_name: str = None,
    strength: float = 0.75,
    num_frames: int = None,
    fps: int = 8
):
    """
    根据视频和文本描述生成新视频
    通过逐帧处理视频实现视频到视频的转换
    
    Args:
        video_path: 输入视频路径
        prompt: 文本描述，说明想要如何修改视频
        output_name: 输出文件名（可选）
        strength: 修改强度 (0.0-1.0)，值越大变化越大
        num_frames: 处理的帧数（None表示处理所有帧）
        fps: 输出视频帧率
    """
    print(f"\n开始根据视频生成新视频...")
    print(f"输入视频: {video_path}")
    print(f"提示词: {prompt}")
    print(f"修改强度: {strength}")
    
    # 获取设备
    device = get_device()
    
    # 加载输入视频
    print("\n正在加载视频...")
    frames = load_video(video_path)
    
    # 如果指定了帧数，进行采样
    if num_frames and len(frames) > num_frames:
        import numpy as np
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
        print(f"采样到 {num_frames} 帧")
    
    print(f"视频帧数: {len(frames)}")
    print(f"原始帧率: {fps}")
    
    # 调整帧大小（模型要求512x512或类似尺寸）
    processed_frames = []
    for i, frame in enumerate(frames):
        frame = resize_image(frame, (512, 512), keep_aspect=True)
        processed_frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(frames)} 帧")
    
    # 加载模型
    print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
    
    # 根据设备选择数据类型
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # 优化：启用注意力切片以节省内存
    try:
        pipe.enable_attention_slicing()
        print("已启用注意力切片（节省内存）")
    except:
        pass
    
    print("模型加载完成！")
    
    # 逐帧处理视频
    print(f"\n正在处理视频（共 {len(processed_frames)} 帧）...")
    print("这可能需要较长时间，请耐心等待...")
    
    output_frames = []
    with torch.no_grad():
        for i, frame in enumerate(processed_frames):
            print(f"处理第 {i + 1}/{len(processed_frames)} 帧...")
            result = pipe(
                prompt=prompt,
                image=frame,
                strength=strength,
                num_inference_steps=50
            )
            output_frames.append(result.images[0])
    
    # 保存视频
    filepath = save_video(output_frames, output_name, "basic_video_to_video", fps=fps)
    print(f"\n✅ 生成完成！")
    return output_frames, filepath


if __name__ == "__main__":
    # 注意：需要先有一个输入视频
    # 示例：将视频转换为油画风格
    # video_path = "examples/videos/input.mp4"  # 替换为你的视频路径
    # generate_video_from_video(
    #     video_path=video_path,
    #     prompt="oil painting style, artistic, detailed brushstrokes",
    #     strength=0.7,
    #     num_frames=30,  # 处理30帧（如果视频很长）
    #     fps=8,
    #     output_name="oil_painting_video"
    # )
    
    print("请先准备一个输入视频，然后取消注释上面的代码并运行")

