"""
下载文本生成视频模型
使用优化的下载工具，支持进度显示、断点续传、错误重试等功能
"""

# 添加项目根目录到路径，以便导入 utils 等模块
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.download_utils import download_model

# 配置参数
REPO_ID = "google/vit-base-patch16-224"
LOCAL_DIR = os.getenv("LOCAL_IMAGE_RECOGNITION_MODEL_PATH", "F:\\modules\\google\\vit-base-patch16-224")

# 可选：只下载必要的文件（取消注释以启用）
# 建议：如果网络不稳定，建议只下载必要文件以减少下载量
# ALLOW_PATTERNS = ["*.safetensors", "*.json", "preprocessor_config.json"]  # 只下载必要文件
# IGNORE_PATTERNS = ["*.bin", "*.ckpt", "*.pt", "*.msgpack", "*training*"]  # 忽略不必要的格式

if __name__ == "__main__":
    try:
        print("\n💡 提示: 如果遇到网络超时问题，建议：")
        print("   1. 使用代理或VPN改善网络连接")
        print("   2. 只下载必要文件（取消注释 allow_patterns）")
        print("   3. 等待网络状况改善后重新运行\n")
        
        # 下载模型（优化后的配置，更好地处理网络超时）
        download_model(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            # allow_patterns=ALLOW_PATTERNS,  # 取消注释以只下载指定文件
            # ignore_patterns=IGNORE_PATTERNS,  # 取消注释以忽略指定文件
            max_workers=2,  # 减少并发数以避免超时（从16降到2）
            resume_download=True,  # 支持断点续传
            check_before_download=True,  # 下载前检查本地是否已存在
            retry_times=10,  # 增加重试次数（从3增到10）
            # timeout=300  # 设置超时时间为300秒
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  下载已取消")
        print("   下次运行时会自动从断点续传")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 尝试使用代理或VPN")
        print("   3. 重新运行脚本（支持断点续传）")
        sys.exit(1)