"""
进阶示例: 文本生成视频（带参数控制）
包含更多参数选项，可以精细控制生成效果
"""

from diffusers import DiffusionPipeline
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import save_video, get_device, set_seed
from config import DEFAULT_VIDEO_STEPS, DEFAULT_VIDEO_FRAMES, DEFAULT_VIDEO_FPS, DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH, DEFAULT_SEED


class AdvancedTextToVideo:
    """高级文本生成视频类"""
    
    def __init__(self, model_name: str = "damo-vilab/text-to-video-ms-1.7b"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.device = get_device()
        self.model_name = model_name
        self.pipe = None
        print(f"初始化视频生成器，使用模型: {model_name}")
    
    def load_model(self):
        """加载模型（延迟加载，只在需要时加载）"""
        if self.pipe is None:
            print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
            print("注意：视频生成模型较大，下载可能需要较长时间")
            
            if self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            )
            self.pipe = self.pipe.to(self.device)
            
            # 优化：启用内存高效注意力
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
        num_inference_steps: int = DEFAULT_VIDEO_STEPS,
        num_frames: int = DEFAULT_VIDEO_FRAMES,
        fps: int = DEFAULT_VIDEO_FPS,
        height: int = DEFAULT_VIDEO_HEIGHT,
        width: int = DEFAULT_VIDEO_WIDTH,
        seed: int = DEFAULT_SEED,
        output_name: str = None
    ):
        """
        生成视频
        
        Args:
            prompt: 正面提示词（描述想要的内容）
            negative_prompt: 负面提示词（描述不想要的内容）
            num_inference_steps: 推理步数（20-100，越多质量越好但越慢）
            num_frames: 视频帧数（8-32，越多越流畅但越慢）
            fps: 帧率（4-12，影响播放速度）
            height: 视频高度（必须是8的倍数）
            width: 视频宽度（必须是8的倍数）
            seed: 随机种子（用于复现结果）
            output_name: 输出文件名
        
        Returns:
            生成的视频帧和文件路径
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
        print(f"  帧数: {num_frames}")
        print(f"  帧率: {fps}")
        print(f"  视频尺寸: {width}x{height}")
        if seed is not None:
            print(f"  随机种子: {seed}")
        print(f"{'='*60}\n")
        
        # 生成视频
        print("正在生成视频（这可能需要几分钟）...")
        print("视频生成比图片生成慢得多，请耐心等待...")
        
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                height=height,
                width=width
            )
            video_frames = result.frames
        
        # 保存视频
        filepath = save_video(video_frames, output_name, "advanced_text_to_video", fps=fps)
        print(f"\n✅ 生成完成！")
        return video_frames, filepath


def main():
    """主函数 - 演示不同参数的效果"""
    generator = AdvancedTextToVideo()
    
    # 示例1: 基础生成
    print("\n【示例1】基础生成")
    generator.generate(
        prompt="a beautiful landscape with mountains and lake, peaceful, sunset",
        output_name="landscape_basic"
    )
    
    # 示例2: 使用负面提示词
    print("\n【示例2】使用负面提示词")
    generator.generate(
        prompt="a cat playing with a ball, cute, cartoon style",
        negative_prompt="blurry, low quality, distorted, ugly",
        output_name="cat_negative"
    )
    
    # 示例3: 更多帧数（更流畅）
    print("\n【示例3】更多帧数（更流畅）")
    generator.generate(
        prompt="ocean waves crashing on the beach, dynamic, cinematic",
        num_frames=24,
        fps=12,
        output_name="ocean_high_fps"
    )
    
    # 示例4: 使用种子复现结果
    print("\n【示例4】使用固定种子（可复现）")
    seed = 42
    generator.generate(
        prompt="a robot walking in a futuristic city, cyberpunk style",
        seed=seed,
        output_name="robot_seed42"
    )


if __name__ == "__main__":
    main()

