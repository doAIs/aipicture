"""
测试视频保存修复
用于验证视频保存功能是否正常工作
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from PIL import Image
import numpy as np
from utils.modules_utils import save_video


def create_test_frames(num_frames=16, size=(256, 256)):
    """
    创建测试用的视频帧
    每一帧都有不同的颜色渐变
    """
    frames = []
    for i in range(num_frames):
        # 创建一个渐变色的测试图像
        # 从蓝色渐变到红色
        r = int(255 * i / num_frames)
        g = 128
        b = int(255 * (1 - i / num_frames))
        
        # 创建纯色图像
        img = Image.new('RGB', size, (r, g, b))
        
        # 添加一些文字标记
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        text = f"Frame {i+1}/{num_frames}"
        
        # 计算文字位置（居中）
        # 使用默认字体
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # 绘制白色文字和黑色描边
        draw.text((x-1, y-1), text, fill=(0, 0, 0))
        draw.text((x+1, y-1), text, fill=(0, 0, 0))
        draw.text((x-1, y+1), text, fill=(0, 0, 0))
        draw.text((x+1, y+1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
        
        frames.append(img)
    
    return frames


def test_video_save():
    """测试视频保存功能"""
    print("="*60)
    print("测试视频保存功能")
    print("="*60)
    
    # 测试1: PIL Image帧
    print("\n" + "="*60)
    print("测试1: 使用PIL Image帧（模拟正常情况）")
    print("="*60)
    
    # 创建测试帧
    print("\n1. 创建测试帧...")
    frames = create_test_frames(num_frames=16, size=(512, 512))
    print(f"   创建了 {len(frames)} 帧测试图像")
    
    # 检查帧数据
    print("\n2. 检查帧数据...")
    first_frame = frames[0]
    print(f"   帧类型: {type(first_frame)}")
    print(f"   帧尺寸: {first_frame.size}")
    print(f"   帧模式: {first_frame.mode}")
    
    frame_array = np.array(first_frame)
    print(f"   数据类型: {frame_array.dtype}")
    print(f"   数据范围: [{frame_array.min()}, {frame_array.max()}]")
    print(f"   数据形状: {frame_array.shape}")
    
    # 保存视频
    print("\n3. 保存视频...")
    filepath = save_video(frames, "test_video_pil", "test", fps=8)
    verify_video(filepath)
    
    # 测试2: 浮点数numpy数组（0-1范围）
    print("\n" + "="*60)
    print("测试2: 使用浮点数numpy数组（0-1范围，模拟扩散模型输出）")
    print("="*60)
    
    # 创建浮点数帧（0-1范围）
    print("\n1. 创建浮点数测试帧...")
    float_frames = []
    for i in range(16):
        # 创建0-1范围的浮点数数组
        r = i / 16
        g = 0.5
        b = 1 - (i / 16)
        img_array = np.ones((512, 512, 3), dtype=np.float32)
        img_array[:, :, 0] = r
        img_array[:, :, 1] = g
        img_array[:, :, 2] = b
        float_frames.append(img_array)
    
    print(f"   创建了 {len(float_frames)} 帧浮点数数组")
    
    # 检查帧数据
    print("\n2. 检查帧数据...")
    first_float_frame = float_frames[0]
    print(f"   帧类型: {type(first_float_frame)}")
    print(f"   数据类型: {first_float_frame.dtype}")
    print(f"   数据范围: [{first_float_frame.min()}, {first_float_frame.max()}]")
    print(f"   数据形状: {first_float_frame.shape}")
    
    # 保存视频
    print("\n3. 保存视频...")
    filepath2 = save_video(float_frames, "test_video_float", "test", fps=8)
    verify_video(filepath2)
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


def verify_video(filepath):
    """验证视频文件"""
    print("\n4. 验证视频文件...")
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"   ✅ 视频文件已创建")
        print(f"   文件路径: {filepath}")
        print(f"   文件大小: {file_size / 1024:.2f} KB")
        
        if file_size < 1000:  # 小于1KB可能有问题
            print(f"   ⚠️  警告：文件大小异常小，可能存在问题")
        else:
            print(f"   ✅ 文件大小正常")
        
        # 尝试读取视频验证
        try:
            import imageio
            reader = imageio.get_reader(filepath)
            meta = reader.get_meta_data()
            print(f"\n5. 读取视频元数据...")
            print(f"   FPS: {meta.get('fps', 'N/A')}")
            print(f"   尺寸: {meta.get('size', 'N/A')}")
            print(f"   时长: {meta.get('duration', 'N/A')}s")
            
            # 读取第一帧验证
            first_frame_read = reader.get_next_data()
            print(f"   第一帧形状: {first_frame_read.shape}")
            print(f"   第一帧数据范围: [{first_frame_read.min()}, {first_frame_read.max()}]")
            
            reader.close()
            print(f"\n   ✅ 视频文件可以正常读取！")
        except Exception as e:
            print(f"\n   ❌ 读取视频时出错: {e}")
    else:
        print(f"   ❌ 视频文件未创建")


if __name__ == "__main__":
    test_video_save()
