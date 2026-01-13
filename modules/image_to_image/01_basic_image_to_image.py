"""
基础示例 2: 图片生成图片（最简单版本）
根据输入图片和文本提示词生成新图片
"""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import save_image, load_image, get_device


def generate_image_from_image(
    image_path: str,
    prompt: str,
    strength: float = 0.75,
    output_name: str = None
):
    """
    根据图片和文本描述生成新图片
    
    Args:
        image_path: 输入图片路径
        prompt: 文本描述，说明想要如何修改图片
        strength: 修改强度 (0.0-1.0)，值越大变化越大
        output_name: 输出文件名（可选）
    """
    print(f"\n开始根据图片生成新图片...")
    print(f"输入图片: {image_path}")
    print(f"提示词: {prompt}")
    print(f"修改强度: {strength}")
    
    # 获取设备
    device = get_device()
    
    # 加载输入图片
    init_image = load_image(image_path)
    print(f"图片尺寸: {init_image.size}")
    
    # 调整图片大小（模型要求512x512或类似尺寸）
    init_image = init_image.resize((512, 512))
    
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
    
    # 生成图片
    print("\n正在生成图片（这可能需要30秒到几分钟）...")
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=50
        ).images[0]
    
    # 保存图片
    filepath = save_image(image, output_name, "basic_image_to_image")
    print(f"\n✅ 生成完成！")
    return image, filepath


if __name__ == "__main__":
    # 注意：需要先有一张输入图片
    # 你可以使用 examples 目录中的图片，或者自己准备一张
    
    # 示例：将图片转换为油画风格
    # image_path = "examples/input.jpg"  # 替换为你的图片路径
    # prompt = "oil painting style, artistic, detailed"
    # generate_image_from_image(image_path, prompt, strength=0.7, output_name="oil_painting")
    
    print("请先准备一张输入图片，然后取消注释上面的代码并运行")

