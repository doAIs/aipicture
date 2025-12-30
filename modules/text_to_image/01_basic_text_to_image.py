"""
基础示例 1: 文本生成图片（最简单版本）
这是最基础的文本生成图片示例，适合初学者理解基本流程
"""

from diffusers import StableDiffusionPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import save_image, get_device


def generate_image_from_text(prompt: str, output_name: str = None):
    """
    根据文本描述生成图片
    
    Args:
        prompt: 文本描述，例如 "a beautiful sunset over the ocean"
        output_name: 输出文件名（可选）
    """
    print(f"\n开始生成图片...")
    print(f"提示词: {prompt}")
    
    # 获取设备
    device = get_device()
    
    # 加载模型（首次运行会自动下载，需要一些时间）
    print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
    
    # 根据设备选择数据类型
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        safety_checker=None,  # 禁用安全检查器以加快速度
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
    
    # 生成图片
    print("\n正在生成图片（这可能需要30秒到几分钟）...")
    with torch.no_grad():
        image = pipe(prompt).images[0]
    
    # 保存图片
    filepath = save_image(image, output_name, "basic_text_to_image")
    print(f"\n✅ 生成完成！")
    return image, filepath


if __name__ == "__main__":
    # 示例1: 生成一张简单的图片
    prompt = "a beautiful sunset over the ocean, peaceful, serene"
    generate_image_from_text(prompt, "sunset_ocean")
    
    # 示例2: 生成另一张图片
    # prompt2 = "a cute cat playing with a ball of yarn, cartoon style"
    # generate_image_from_text(prompt2, "cat_yarn")

