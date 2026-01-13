"""
基础示例: 图片生成视频（最简单版本）
根据输入图片和文本提示词生成视频
"""

from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import save_video, load_image, get_device, resize_image


def generate_video_from_image(
    image_path: str,
    prompt: str = None,
    output_name: str = None,
    num_frames: int = 14,
    fps: int = 7,
    motion_bucket_id: int = 127
):
    """
    根据图片和文本描述生成视频
    
    Args:
        image_path: 输入图片路径
        prompt: 文本描述（可选，用于引导生成）
        output_name: 输出文件名（可选）
        num_frames: 视频帧数（默认14帧）
        fps: 帧率（默认7fps）
        motion_bucket_id: 运动强度（1-255，越大运动越剧烈）
    """
    print(f"\n开始根据图片生成视频...")
    print(f"输入图片: {image_path}")
    if prompt:
        print(f"提示词: {prompt}")
    print(f"帧数: {num_frames}, 帧率: {fps}")
    print(f"运动强度: {motion_bucket_id}")
    
    # 获取设备
    device = get_device()
    
    # 加载输入图片
    init_image = load_image(image_path)
    print(f"图片尺寸: {init_image.size}")
    
    # 调整图片大小（模型要求特定尺寸）
    # Stable Video Diffusion 通常需要 1024x576 或类似尺寸
    init_image = resize_image(init_image, (1024, 576), keep_aspect=True)
    print(f"调整后尺寸: {init_image.size}")
    
    # 加载模型
    print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
    print("注意：视频生成模型较大，下载可能需要较长时间")
    
    # 根据设备选择数据类型
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
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
                image=init_image,
                decode_chunk_size=2,  # 分块解码以节省内存
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=0.02
            ).frames
        
        # 保存视频
        filepath = save_video(video_frames, output_name, "basic_image_to_video", fps=fps)
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
    # 注意：需要先有一张输入图片
    # 示例：根据图片生成视频
    # image_path = "examples/images/input.jpg"  # 替换为你的图片路径
    # generate_video_from_image(
    #     image_path=image_path,
    #     prompt="smooth camera movement, cinematic",
    #     num_frames=14,
    #     fps=7,
    #     output_name="image_to_video_demo"
    # )
    
    print("请先准备一张输入图片，然后取消注释上面的代码并运行")

