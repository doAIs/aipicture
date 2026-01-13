"""
进阶示例: 图片生成视频（带参数控制）
包含更多参数选项，可以精细控制视频生成效果
"""

from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.modules_utils import save_video, load_image, get_device, set_seed, resize_image
from config.modules_config import DEFAULT_VIDEO_FRAMES, DEFAULT_VIDEO_FPS, DEFAULT_SEED


class AdvancedImageToVideo:
    """高级图片生成视频类"""
    
    def __init__(self, model_name: str = "stabilityai/stable-video-diffusion-img2vid"):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
        """
        self.device = get_device()
        self.model_name = model_name
        self.pipe = None
        print(f"初始化图片生成视频器，使用模型: {model_name}")
    
    def load_model(self):
        """加载模型（延迟加载）"""
        if self.pipe is None:
            print("\n正在加载模型（首次运行需要下载，请耐心等待）...")
            print("注意：视频生成模型较大，下载可能需要较长时间")
            
            if self.device == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
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
        num_frames: int = DEFAULT_VIDEO_FRAMES,
        fps: int = DEFAULT_VIDEO_FPS,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 2,
        seed: int = DEFAULT_SEED,
        output_name: str = None
    ):
        """
        根据图片生成视频
        
        Args:
            image_path: 输入图片路径
            num_frames: 视频帧数（14-25，越多越流畅但越慢）
            fps: 帧率（4-12，影响播放速度）
            motion_bucket_id: 运动强度（1-255）
                            - 1-50: 静态，几乎没有运动
                            - 51-127: 轻微运动
                            - 128-200: 中等运动
                            - 201-255: 剧烈运动
            noise_aug_strength: 噪声增强强度（0.0-1.0），影响生成多样性
            decode_chunk_size: 解码块大小（1-8），越小越省内存但越慢
            seed: 随机种子
            output_name: 输出文件名
        
        Returns:
            生成的视频帧和文件路径
        """
        # 确保模型已加载
        self.load_model()
        
        # 加载输入图片
        init_image = load_image(image_path)
        original_size = init_image.size
        print(f"输入图片尺寸: {original_size}")
        
        # 调整图片大小（Stable Video Diffusion 需要特定尺寸）
        init_image = resize_image(init_image, (1024, 576), keep_aspect=True)
        print(f"调整后尺寸: {init_image.size}")
        
        # 设置随机种子
        if seed is not None:
            set_seed(seed)
        
        print(f"\n{'='*60}")
        print(f"生成参数:")
        print(f"  输入图片: {image_path}")
        print(f"  帧数: {num_frames}")
        print(f"  帧率: {fps}")
        print(f"  运动强度: {motion_bucket_id}")
        print(f"  噪声增强: {noise_aug_strength}")
        if seed is not None:
            print(f"  随机种子: {seed}")
        print(f"{'='*60}\n")
        
        # 生成视频
        print("正在生成视频...")
        print("视频生成比图片生成慢得多，请耐心等待...")
        
        with torch.no_grad():
            result = self.pipe(
                image=init_image,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size
            )
            video_frames = result.frames
        
        # 保存视频
        filepath = save_video(video_frames, output_name, "advanced_image_to_video", fps=fps)
        print(f"\n✅ 生成完成！")
        return video_frames, filepath


def main():
    """主函数 - 演示不同参数的效果"""
    generator = AdvancedImageToVideo()
    
    # 注意：需要准备输入图片
    # 示例1: 静态场景（低运动强度）
    print("\n【示例1】静态场景（低运动强度）")
    # generator.generate(
    #     image_path="examples/images/landscape.jpg",
    #     motion_bucket_id=50,  # 低运动强度
    #     num_frames=14,
    #     output_name="static_scene"
    # )
    
    # 示例2: 动态场景（高运动强度）
    print("\n【示例2】动态场景（高运动强度）")
    # generator.generate(
    #     image_path="examples/images/action.jpg",
    #     motion_bucket_id=200,  # 高运动强度
    #     num_frames=21,
    #     fps=10,
    #     output_name="dynamic_scene"
    # )
    
    # 示例3: 使用种子复现结果
    print("\n【示例3】使用固定种子（可复现）")
    # seed = 42
    # generator.generate(
    #     image_path="examples/images/input.jpg",
    #     seed=seed,
    #     output_name="reproducible_video"
    # )
    
    print("\n请准备输入图片，然后取消注释上面的代码并运行")


if __name__ == "__main__":
    main()

