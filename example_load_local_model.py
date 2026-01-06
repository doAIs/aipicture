"""
示例：如何使用本地下载的 Stable Diffusion 模型文件

本示例展示如何加载本地下载的 .safetensors 或 .ckpt 模型文件
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from modules.text_to_image import generate_image_from_text

if __name__ == "__main__":
    # 方式1: 使用本地模型文件（.safetensors 或 .ckpt）
    # 请将下面的路径替换为您实际的模型文件路径
    local_model_path = "E:\GIT_AI\modules\text_to_image\v1-5-pruned.safetensors"  # 或者使用完整路径，如 "E:/models/v1-5-pruned.safetensors"
    
    prompt = "a beautiful sunset over the ocean, peaceful, serene"
    
    # 使用本地模型生成图片
    print("=" * 60)
    print("使用本地模型文件生成图片")
    print("=" * 60)
    generate_image_from_text(
        prompt=prompt,
        output_name="sunset_local_model",
        model_path=local_model_path  # 指定本地模型路径
    )
    
    # 方式2: 不指定 model_path，使用默认的在线模型
    # generate_image_from_text(prompt, "sunset_online")

