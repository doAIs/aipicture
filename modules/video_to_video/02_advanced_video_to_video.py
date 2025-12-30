"""
进阶示例: 视频生成视频（带参数控制）
包含更多参数选项，可以精细控制视频转换效果
"""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import load_video, save_video, get_device, set_seed, resize_image
from config import DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_SEED
import numpy as np


class AdvancedVideoToVideo:
    """高级视频生成视频类"""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.device = get_device()
        self.model_name = model_name
        self.pipe = None
        print(f"初始化视频生成视频器，使用模型: {model_name}")
    
    def load_model(self):
        """加载模型（延迟加载）"""
        if self.pipe is None:
            print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
            
            if self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
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
        video_path: str,
        prompt: str,
        negative_prompt: str = None,
        strength: float = 0.75,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_frames: int = None,
        fps: int = 8,
        frame_size: tuple = (512, 512),
        seed: int = DEFAULT_SEED,
        output_name: str = None
    ):
        """
        根据视频生成新视频
        
        Args:
            video_path: 输入视频路径
            prompt: 正面提示词
            negative_prompt: 负面提示词
            strength: 修改强度 (0.0-1.0)
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            num_frames: 处理的帧数（None表示处理所有帧）
            fps: 输出视频帧率
            frame_size: 帧尺寸 (width, height)
            seed: 随机种子
            output_name: 输出文件名
        
        Returns:
            生成的视频帧和文件路径
        """
        # 确保模型已加载
        self.load_model()
        
        # 加载输入视频
        print("\n正在加载视频...")
        frames = load_video(video_path)
        
        # 如果指定了帧数，进行采样
        if num_frames and len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
            print(f"采样到 {num_frames} 帧")
        
        print(f"视频帧数: {len(frames)}")
        
        # 调整帧大小
        processed_frames = []
        for i, frame in enumerate(frames):
            frame = resize_image(frame, frame_size, keep_aspect=True)
            processed_frames.append(frame)
        
        # 设置随机种子
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"生成参数:")
        print(f"  输入视频: {video_path}")
        print(f"  提示词: {prompt}")
        if negative_prompt:
            print(f"  负面提示词: {negative_prompt}")
        print(f"  修改强度: {strength}")
        print(f"  推理步数: {num_inference_steps}")
        print(f"  引导强度: {guidance_scale}")
        print(f"  处理帧数: {len(processed_frames)}")
        print(f"  输出帧率: {fps}")
        if seed is not None:
            print(f"  随机种子: {seed}")
        print(f"{'='*60}\n")
        
        # 逐帧处理视频
        print(f"正在处理视频（共 {len(processed_frames)} 帧）...")
        print("这可能需要较长时间，请耐心等待...")
        
        output_frames = []
        with torch.no_grad():
            for i, frame in enumerate(processed_frames):
                print(f"处理第 {i + 1}/{len(processed_frames)} 帧...")
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=frame,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                output_frames.append(result.images[0])
        
        # 保存视频
        filepath = save_video(output_frames, output_name, "advanced_video_to_video", fps=fps)
        print(f"\n✅ 生成完成！")
        return output_frames, filepath


def main():
    """主函数 - 演示不同参数的效果"""
    generator = AdvancedVideoToVideo()
    
    # 注意：需要准备输入视频
    # 示例1: 风格转换（低强度，保持原视频结构）
    print("\n【示例1】风格转换（低强度）")
    # generator.generate(
    #     video_path="examples/videos/input.mp4",
    #     prompt="oil painting style, artistic, detailed brushstrokes",
    #     strength=0.5,  # 低强度，保持原视频结构
    #     num_frames=30,
    #     output_name="oil_painting_low"
    # )
    
    # 示例2: 风格转换（高强度，更多变化）
    print("\n【示例2】风格转换（高强度）")
    # generator.generate(
    #     video_path="examples/videos/input.mp4",
    #     prompt="anime style, vibrant colors, detailed",
    #     negative_prompt="blurry, low quality, distorted",
    #     strength=0.8,  # 高强度，更多变化
    #     num_frames=30,
    #     output_name="anime_high"
    # )
    
    # 示例3: 季节转换
    print("\n【示例3】季节转换")
    # generator.generate(
    #     video_path="examples/videos/summer.mp4",
    #     prompt="winter scene, snow, cold atmosphere, peaceful",
    #     strength=0.7,
    #     num_frames=30,
    #     output_name="winter_scene"
    # )
    
    print("\n请准备输入视频，然后取消注释上面的代码并运行")


if __name__ == "__main__":
    main()

