from modelscope import snapshot_download
import os
import sys


def main():
    # 配置参数
    REPO_ID = "google/vit-base-patch16-224"
    LOCAL_DIR = r"F:\modules\google\vit-base-patch16-224"

    print("============================================================")
    print("开始下载模型（ModelScope 国内源）")
    print("============================================================")
    print(f"模型仓库: {REPO_ID}")
    print(f"保存位置: {LOCAL_DIR}")
    print("============================================================")

    try:
        # 确保目标目录存在
        os.makedirs(LOCAL_DIR, exist_ok=True)

        # 下载模型
        model_dir = snapshot_download(
            model_id=REPO_ID,
            cache_dir=LOCAL_DIR
        )

        print("\n============================================================")
        print("✅ 下载成功！")
        print(f"模型路径: {model_dir}")
        print("============================================================")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间")
        print("3. 尝试使用 pip install -U modelscope 更新 modelscope")
        sys.exit(1)


if __name__ == "__main__":
    main()
