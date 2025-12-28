"""
综合示例 5: 完整功能演示
展示文本生成图片和图片生成图片的完整工作流程
包含多个实际应用场景
"""

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from utils import save_image, load_image, get_device, set_seed
from config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH
)
import os


class ComprehensiveImageGenerator:
    """综合图片生成器 - 整合所有功能"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.device = get_device()
        self.model_name = model_name
        self.text_pipe = None
        self.img2img_pipe = None
        print(f"初始化综合图片生成器，使用模型: {model_name}")
    
    def load_text_to_image_model(self):
        """加载文本生成图片模型"""
        if self.text_pipe is None:
            print("\n正在加载文本生成图片模型...")
            self.text_pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.text_pipe = self.text_pipe.to(self.device)
            try:
                self.text_pipe.enable_attention_slicing()
            except:
                pass
            print("文本生成图片模型加载完成！")
    
    def load_image_to_image_model(self):
        """加载图片生成图片模型"""
        if self.img2img_pipe is None:
            print("\n正在加载图片生成图片模型...")
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            try:
                self.img2img_pipe.enable_attention_slicing()
            except:
                pass
            print("图片生成图片模型加载完成！")
    
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        seed: int = None,
        output_name: str = None
    ):
        """
        文本生成图片
        
        Returns:
            生成的图片和文件路径
        """
        self.load_text_to_image_model()
        
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"文本生成图片")
        print(f"  提示词: {prompt}")
        if negative_prompt:
            print(f"  负面提示词: {negative_prompt}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            result = self.text_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            image = result.images[0]
        
        filepath = save_image(image, output_name, "comprehensive")
        print(f"✅ 生成完成！")
        return image, filepath
    
    def image_to_image(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = None,
        strength: float = 0.75,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int = None,
        output_name: str = None
    ):
        """
        图片生成图片
        
        Returns:
            生成的图片和文件路径
        """
        self.load_image_to_image_model()
        
        init_image = load_image(image_path)
        max_size = 512
        if max(init_image.size) > max_size:
            ratio = max_size / max(init_image.size)
            new_size = (int(init_image.size[0] * ratio), int(init_image.size[1] * ratio))
            init_image = init_image.resize(new_size, Image.Resampling.LANCZOS)
        
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"图片生成图片")
        print(f"  输入图片: {image_path}")
        print(f"  提示词: {prompt}")
        print(f"  修改强度: {strength}")
        print(f"{'='*60}\n")
        
        with torch.no_grad():
            result = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            image = result.images[0]
        
        filepath = save_image(image, output_name, "comprehensive")
        print(f"✅ 生成完成！")
        return image, filepath
    
    def workflow_example(self):
        """
        完整工作流示例：
        1. 先用文本生成一张基础图片
        2. 再用这张图片进行风格转换
        """
        print("\n" + "="*60)
        print("完整工作流示例")
        print("="*60)
        
        # 步骤1: 文本生成图片
        print("\n【步骤1】文本生成图片")
        prompt1 = "a beautiful mountain landscape, lake, sunset, peaceful"
        image1, path1 = self.text_to_image(
            prompt=prompt1,
            negative_prompt="blurry, low quality, distorted",
            num_inference_steps=50,
            output_name="workflow_step1_landscape"
        )
        
        # 步骤2: 图片生成图片（风格转换）
        print("\n【步骤2】图片生成图片（风格转换）")
        image2, path2 = self.image_to_image(
            image_path=path1,
            prompt="oil painting style, artistic, detailed brushstrokes, Van Gogh style",
            strength=0.7,
            output_name="workflow_step2_painting"
        )
        
        print("\n✅ 完整工作流执行完成！")
        print(f"  原始图片: {path1}")
        print(f"  转换后图片: {path2}")
        
        return image1, image2, path1, path2


def demo_scenarios():
    """演示多个实际应用场景"""
    generator = ComprehensiveImageGenerator()
    
    scenarios = [
        {
            "name": "场景1: 创意设计 - 生成概念图",
            "type": "text_to_image",
            "prompt": "a futuristic cityscape at night, neon lights, cyberpunk style, detailed, 4k",
            "negative_prompt": "blurry, low quality, distorted",
            "output_name": "scenario1_futuristic_city"
        },
        {
            "name": "场景2: 艺术创作 - 生成艺术作品",
            "type": "text_to_image",
            "prompt": "a serene Japanese garden, cherry blossoms, traditional architecture, peaceful, watercolor style",
            "negative_prompt": "ugly, distorted, low quality",
            "output_name": "scenario2_japanese_garden"
        },
        {
            "name": "场景3: 产品设计 - 生成产品概念",
            "type": "text_to_image",
            "prompt": "a modern minimalist chair, white background, studio lighting, product photography, high quality",
            "negative_prompt": "cluttered, low quality, blurry",
            "output_name": "scenario3_product_chair"
        }
    ]
    
    print("\n" + "="*60)
    print("多场景演示")
    print("="*60)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{scenario['name']}")
        if scenario['type'] == 'text_to_image':
            generator.text_to_image(
                prompt=scenario['prompt'],
                negative_prompt=scenario.get('negative_prompt'),
                output_name=scenario['output_name']
            )
    
    # 如果有输入图片，演示图片生成图片场景
    example_dir = "examples"
    if os.path.exists(example_dir):
        image_files = [f for f in os.listdir(example_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            print("\n" + "="*60)
            print("图片生成图片场景演示")
            print("="*60)
            
            image_path = os.path.join(example_dir, image_files[0])
            print(f"\n使用输入图片: {image_path}")
            
            # 场景4: 风格转换
            print("\n【场景4】风格转换")
            generator.image_to_image(
                image_path=image_path,
                prompt="anime style, vibrant colors, detailed, studio ghibli",
                strength=0.75,
                output_name="scenario4_anime_style"
            )
            
            # 场景5: 季节变换
            print("\n【场景5】季节变换")
            generator.image_to_image(
                image_path=image_path,
                prompt="winter scene, snow, cold atmosphere, peaceful",
                strength=0.7,
                output_name="scenario5_winter"
            )


def main():
    """主函数"""
    print("="*60)
    print("AI 图片生成 - 综合示例")
    print("="*60)
    print("\n本示例展示完整的功能和工作流程")
    print("包括：文本生成图片、图片生成图片、完整工作流")
    
    generator = ComprehensiveImageGenerator()
    
    # 示例1: 基础文本生成图片
    print("\n" + "="*60)
    print("示例1: 基础文本生成图片")
    print("="*60)
    generator.text_to_image(
        prompt="a cute cat playing with a ball of yarn, cartoon style, colorful, detailed",
        negative_prompt="blurry, low quality, distorted",
        output_name="example1_cat"
    )
    
    # 示例2: 完整工作流
    print("\n" + "="*60)
    print("示例2: 完整工作流（文本生成 -> 风格转换）")
    print("="*60)
    generator.workflow_example()
    
    # 示例3: 多场景演示
    print("\n" + "="*60)
    print("示例3: 多场景演示")
    print("="*60)
    demo_scenarios()
    
    print("\n" + "="*60)
    print("所有示例执行完成！")
    print("="*60)
    print("\n生成的图片保存在 outputs/comprehensive/ 目录下")


if __name__ == "__main__":
    main()

