# download_google_vit_image_recognition.py
import sys

from huggingface_hub import snapshot_download

# 配置
REPO_ID = "google/vit-base-patch16-224"
LOCAL_DIR = r"F:\modules\google\vit-base-patch16-224"

print("============================================================")
print("开始下载模型")
print("============================================================")
print(f"模型仓库: {REPO_ID}")
print(f"保存位置: {LOCAL_DIR}")
print("============================================================")

# 关键参数说明：
# - ignore_patterns: 跳过不需要的框架文件（节省时间/空间）
# - resume_download: 自动断点续传
# - max_workers: 并发数（2～4 足够）
# - token: 如需私有模型才填（公开模型不用）
# - endpoint: 强制使用官方源（避免国内镜像）
if __name__ == "__main__":
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            # local_dir_use_symlinks=False,  # 直接复制文件
            resume_download=True,  # 断点续传
            max_workers=16,
            # 只下载 PyTorch 用户需要的文件（跳过 Flax/TensorFlow）;
            # ignore_patterns=["*.msgpack", "*.h5", "tf_model.*"],
            # 强制使用官方 Hugging Face 源（绕过 hf-mirror）
            endpoint="https://hf-mirror.com"
        )
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)
