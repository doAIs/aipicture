"""
进阶示例 3: 文本生成图片（带参数控制）
包含更多参数选项，可以精细控制生成效果
"""

from diffusers import StableDiffusionPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import save_image, get_device, set_seed
from config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SEED,
    LOCAL_MODEL_PATH
)


class AdvancedTextToImage:
    """高级文本生成图片类"""
    
    def __init__(self, model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", local_model_path: str = None):
        """
        初始化生成器
        
        Args:
            model_name: 在线模型名称（HuggingFace ID）
            local_model_path: 本地模型路径（可选）
                            - 如果为 None，则从配置文件 config.LOCAL_MODEL_PATH 读取
                            - 如果为 "" 或空字符串，则禁用本地模型，仅使用在线模型
                            - 如果指定路径，则使用指定的路径
        """
        self.device = get_device()
        self.model_name = model_name
        
        # 确定本地模型路径的优先级：
        # 1. 如果传入了 local_model_path 参数，使用传入的值（空字符串表示禁用）
        # 2. 否则从配置文件读取 LOCAL_MODEL_PATH
        if local_model_path is not None:
            self.local_model_path = local_model_path if local_model_path else None
        else:
            # 从配置文件读取，如果配置为 None 或空字符串，则禁用本地模型
            self.local_model_path = LOCAL_MODEL_PATH if LOCAL_MODEL_PATH else None
        
        self.pipe = None
        print(f"初始化生成器，在线模型: {model_name}")
        if self.local_model_path:
            print(f"本地模型路径: {self.local_model_path}")
        else:
            print("本地模型: 已禁用（仅使用在线模型）")
    
    def load_model(self):
        """加载模型（延迟加载，只在需要时加载）
        优先加载本地离线模型，如果本地模型不存在则加载在线模型
        """
        if self.pipe is None:
            from utils import load_model_from_local_file, load_model_with_fallback
            
            # 准备模型加载参数
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False
            }
            
            # 优先尝试加载本地模型（如果已配置）
            if self.local_model_path and os.path.exists(self.local_model_path):
                print(f"\n✅ 检测到本地模型路径: {self.local_model_path}")
                print("   优先使用本地离线模型...")
                try:
                    self.pipe = load_model_from_local_file(
                        StableDiffusionPipeline,
                        self.local_model_path,
                        **model_kwargs
                    )
                    self.pipe = self.pipe.to(self.device)
                    print("✅ 本地模型加载成功！")
                except Exception as e:
                    print(f"⚠️  本地模型加载失败: {e}")
                    print(f"   回退到在线模型: {self.model_name}")
                    # 本地模型加载失败，回退到在线模型
                    self.pipe = load_model_with_fallback(
                        StableDiffusionPipeline,
                        self.model_name,
                        **model_kwargs
                    )
                    self.pipe = self.pipe.to(self.device)
            elif self.local_model_path:
                # 本地模型路径已配置但不存在，直接使用在线模型
                print(f"\n⚠️  本地模型路径不存在: {self.local_model_path}")
                print(f"   使用在线模型: {self.model_name}")
                self.pipe = load_model_with_fallback(
                    StableDiffusionPipeline,
                    self.model_name,
                    **model_kwargs
                )
                self.pipe = self.pipe.to(self.device)
            else:
                # 本地模型未配置，直接使用在线模型
                print(f"\n📡 使用在线模型: {self.model_name}")
                self.pipe = load_model_with_fallback(
                    StableDiffusionPipeline,
                    self.model_name,
                    **model_kwargs
                )
                self.pipe = self.pipe.to(self.device)
            
            # 优化：启用内存高效注意力（如果支持）
            try:
                self.pipe.enable_attention_slicing()
                print("已启用注意力切片（节省内存）")
            except:
                pass
            
            print("模型加载完成！")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        seed: int = DEFAULT_SEED,
        output_name: str = None
    ):
        """
        生成图片
        
        Args:
            prompt: 正面提示词（描述想要的内容）
            negative_prompt: 负面提示词（描述不想要的内容）
            num_inference_steps: 推理步数（20-100，越多质量越好但越慢）
            guidance_scale: 引导强度（1-20，越高越遵循提示词）
            height: 图片高度（必须是8的倍数）
            width: 图片宽度（必须是8的倍数）
            seed: 随机种子（用于复现结果）
            output_name: 输出文件名
        
        Returns:
            生成的图片和文件路径
        """
        # 确保模型已加载
        self.load_model()
        
        # 设置随机种子
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"生成参数:")
        print(f"  提示词: {prompt}")
        if negative_prompt:
            print(f"  负面提示词: {negative_prompt}")
        print(f"  推理步数: {num_inference_steps}")
        print(f"  引导强度: {guidance_scale}")
        print(f"  图片尺寸: {width}x{height}")
        if seed is not None:
            print(f"  随机种子: {seed}")
        print(f"{'='*60}\n")
        
        # 生成图片
        print("正在生成图片...")
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            image = result.images[0]
        
        # 保存图片
        filepath = save_image(image, output_name, "advanced_text_to_image")
        print(f"\n✅ 生成完成！")
        return image, filepath


def main():
    """主函数 - 演示不同参数的效果"""
    generator = AdvancedTextToImage()
    
    # 示例1: 基础生成
    print("\n【示例1】基础生成")
    generator.generate(
        prompt="a majestic lion standing on a rock, sunset, photorealistic",
        output_name="lion_basic"
    )
    
    # 示例2: 使用负面提示词
    print("\n【示例2】使用负面提示词（排除不想要的内容）")
    generator.generate(
        prompt="a beautiful landscape, mountains, lake, peaceful",
        negative_prompt="blurry, low quality, distorted, ugly",
        output_name="landscape_negative"
    )
    
    # 示例3: 高分辨率生成
    print("\n【示例3】高分辨率生成（768x768）")
    generator.generate(
        prompt="a futuristic city at night, neon lights, cyberpunk style",
        height=768,
        width=768,
        num_inference_steps=60,  # 高分辨率需要更多步数
        output_name="city_highres"
    )
    
    # 示例4: 使用种子复现结果
    print("\n【示例4】使用固定种子（可复现）")
    seed = 42
    generator.generate(
        prompt="a cute robot, cartoon style, colorful",
        seed=seed,
        output_name="robot_seed42"
    )
    
    # 再次使用相同种子，应该得到相同结果
    generator.generate(
        prompt="a cute robot, cartoon style, colorful",
        seed=seed,
        output_name="robot_seed42_repeat"
    )


if __name__ == "__main__":
    main()

