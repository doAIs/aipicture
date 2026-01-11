import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键！启用镜像

from huggingface_hub import snapshot_download

# 确保目标目录存在
local_dir = "F:\\modules\\sd"
os.makedirs(local_dir, exist_ok=True)

print(f"开始下载模型到: {local_dir}")
print("模型仓库: stable-diffusion-v1-5/stable-diffusion-v1-5")

try:
    # 下载 stable-diffusion-v1-5/stable-diffusion-v1-5 模型
    # 注意：如果要去除过滤，可以注释掉 allow_patterns 和 ignore_patterns
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir=local_dir,
        # allow_patterns=["*.safetensors", "*.json", "tokenizer/*", "scheduler/*"],  # 只下载指定文件
        # ignore_patterns=["*.ckpt", "*.pt", "*fp16*", "*training*"],  # 忽略的文件类型
        resume_download=True,
        max_workers=16,  # 多线程加速
    )
    print(f"模型下载完成！保存位置: {local_dir}")
except Exception as e:
    print(f"下载过程中出现错误: {e}")
    raise