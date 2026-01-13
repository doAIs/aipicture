"""
进阶示例 4: 图片生成图片（带参数控制）
包含更多参数选项，可以精细控制图片转换效果
"""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from utils.modules_utils import save_image, load_image, get_device, set_seed
from config.modules_config import DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_SEED


class AdvancedImageToImage:
    """高级图片生成图片类"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.device = get_device()
        self.model_name = model_name
        self.pipe = None
        print(f"初始化生成器，使用模型: {model_name}")
    
    def load_model(self):
        """加载模型（延迟加载）"""
        if self.pipe is None:
            print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe = self.pipe.to(self.device)
            
            try:
                self.pipe.enable_attention_slicing()
                print("已启用注意力切片（节省内存）")
            except:
                pass
            
            print("模型加载完成！")
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = None,
        strength: float = 0.75,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int = DEFAULT_SEED,
        output_name: str = None
    ):
        """
        根据图片生成新图片
        
        Args:
            image_path: 输入图片路径
            prompt: 正面提示词
            negative_prompt: 负面提示词
            strength: 修改强度 (0.0-1.0)
                     - 0.0: 几乎不改变原图
                     - 0.5: 中等改变
                     - 1.0: 完全重新生成（类似text-to-image）
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            seed: 随机种子
            output_name: 输出文件名
        
        Returns:
            生成的图片和文件路径
        """
        # 确保模型已加载
        self.load_model()
        
        # 加载输入图片
        init_image = load_image(image_path)
        original_size = init_image.size
        print(f"输入图片尺寸: {original_size}")
        
        # 调整图片大小（建议512x512或768x768）
        # 保持宽高比
        max_size = 512
        if max(init_image.size) > max_size:
            ratio = max_size / max(init_image.size)
            new_size = (int(init_image.size[0] * ratio), int(init_image.size[1] * ratio))
            init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"调整后尺寸: {init_image.size}")
        
        # 设置随机种子
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"生成参数:")
        print(f"  输入图片: {image_path}")
        print(f"  提示词: {prompt}")
        if negative_prompt:
            print(f"  负面提示词: {negative_prompt}")
        print(f"  修改强度: {strength}")
        print(f"  推理步数: {num_inference_steps}")
        print(f"  引导强度: {guidance_scale}")
        if seed is not None:
            print(f"  随机种子: {seed}")
        print(f"{'='*60}\n")
        
        # 生成图片
        print("正在生成图片...")
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            image = result.images[0]
        
        # 保存图片
        filepath = save_image(image, output_name, "advanced_image_to_image")
        print(f"\n✅ 生成完成！")
        return image, filepath


def main():
    """主函数 - 演示不同参数的效果"""
    generator = AdvancedImageToImage()
    
    # 注意：需要准备输入图片
    # 示例1: 风格转换（低强度，保持原图结构）
    print("\n【示例1】风格转换（低强度）")
    # generator.generate(
    #     image_path="examples/input.jpg",
    #     prompt="oil painting style, artistic, detailed brushstrokes",
    #     strength=0.5,  # 低强度，保持原图结构
    #     output_name="oil_painting_low"
    # )
    
    # 示例2: 风格转换（高强度，更多变化）
    print("\n【示例2】风格转换（高强度）")
    # generator.generate(
    #     image_path="examples/input.jpg",
    #     prompt="anime style, vibrant colors, detailed",
    #     strength=0.8,  # 高强度，更多变化
    #     output_name="anime_high"
    # )
    
    # 示例3: 添加元素
    print("\n【示例3】添加元素到图片")
    # generator.generate(
    #     image_path="examples/landscape.jpg",
    #     prompt="add a beautiful rainbow in the sky, photorealistic",
    #     negative_prompt="distorted, blurry, low quality",
    #     strength=0.6,
    #     output_name="landscape_rainbow"
    # )
    
    # 示例4: 季节转换
    print("\n【示例4】季节转换")
    # generator.generate(
    #     image_path="examples/summer.jpg",
    #     prompt="winter scene, snow, cold atmosphere",
    #     strength=0.7,
    #     output_name="winter_scene"
    # )
    
    print("\n请准备输入图片，然后取消注释上面的代码并运行")


if __name__ == "__main__":
    main()

