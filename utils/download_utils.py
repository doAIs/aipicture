"""
HuggingFace 模型下载工具
提供统一的模型下载接口，支持进度显示、断点续传、错误重试等功能
"""

import os
import sys
import time
from typing import Optional, List, Callable
from pathlib import Path

# 设置镜像端点（优先使用国内镜像）
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HfHubHTTPError


# 注意：huggingface_hub 的 snapshot_download 会自动显示下载进度
# 这里保留一个简单的工具类用于未来扩展
class DownloadProgress:
    """下载进度工具类（预留用于未来扩展）"""
    pass


def check_model_exists(repo_id: str) -> bool:
    """
    检查模型是否存在
    
    Args:
        repo_id: 模型仓库ID
    
    Returns:
        是否存在
    """
    try:
        api = HfApi()
        api.model_info(repo_id)
        return True
    except Exception:
        return False


def check_local_model_complete(local_dir: str) -> bool:
    """
    检查本地模型是否完整（简单检查，检查目录是否存在且非空）
    
    Args:
        local_dir: 本地目录路径
    
    Returns:
        是否完整
    """
    if not os.path.exists(local_dir):
        return False
    
    # 检查目录是否为空
    if not os.listdir(local_dir):
        return False
    
    # 检查是否有必要的文件（至少有一个文件）
    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    return len(files) > 0


def download_model(
    repo_id: str,
    local_dir: str,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
    max_workers: int = 16,
    resume_download: bool = True,
    check_before_download: bool = True,
    show_progress: bool = True,
    retry_times: int = 3
) -> str:
    """
    下载 HuggingFace 模型
    
    Args:
        repo_id: 模型仓库ID（如 "stable-diffusion-v1-5/stable-diffusion-v1-5"）
        local_dir: 本地保存目录
        allow_patterns: 允许下载的文件模式（如 ["*.safetensors", "*.json"]）
        ignore_patterns: 忽略的文件模式（如 ["*.ckpt", "*.pt"]）
        max_workers: 最大并发下载线程数
        resume_download: 是否支持断点续传
        check_before_download: 下载前是否检查本地是否已存在
        show_progress: 是否显示进度
        retry_times: 重试次数
    
    Returns:
        本地模型目录路径
    
    Raises:
        FileNotFoundError: 模型不存在
        OSError: 下载失败
    """
    # 确保目标目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    # 检查本地模型是否已存在
    if check_before_download and check_local_model_complete(local_dir):
        print(f"✅ 本地模型已存在: {local_dir}")
        print("   跳过下载（如需重新下载，请删除该目录）")
        return local_dir
    
    # 检查模型是否存在
    print(f"\n{'='*60}")
    print(f"开始下载模型")
    print(f"{'='*60}")
    print(f"模型仓库: {repo_id}")
    print(f"保存位置: {local_dir}")
    
    if not check_model_exists(repo_id):
        raise FileNotFoundError(f"模型不存在: {repo_id}")
    
    # 显示下载配置
    if allow_patterns:
        print(f"允许文件: {', '.join(allow_patterns)}")
    if ignore_patterns:
        print(f"忽略文件: {', '.join(ignore_patterns)}")
    print(f"并发线程: {max_workers}")
    print(f"断点续传: {'是' if resume_download else '否'}")
    print(f"{'='*60}\n")
    
    # 下载参数
    download_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "resume_download": resume_download,
        "max_workers": max_workers,
    }
    
    if allow_patterns:
        download_kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        download_kwargs["ignore_patterns"] = ignore_patterns
    
    # 记录开始时间
    start_time = time.time()
    
    # 重试下载
    last_error = None
    for attempt in range(1, retry_times + 1):
        try:
            if attempt > 1:
                print(f"\n⚠️  第 {attempt} 次尝试下载...")
            
            # 执行下载（huggingface_hub 会自动显示进度）
            snapshot_download(**download_kwargs)
            
            # 下载成功
            elapsed_time = time.time() - start_time
            print(f"\n✅ 模型下载完成！")
            print(f"   保存位置: {local_dir}")
            print(f"   耗时: {_format_time(elapsed_time)}")
            return local_dir
            
        except HfHubHTTPError as e:
            last_error = e
            if e.status_code == 404:
                raise FileNotFoundError(f"模型不存在: {repo_id}") from e
            elif e.status_code == 401:
                raise PermissionError(f"无权限访问模型: {repo_id}（可能需要登录）") from e
            elif attempt < retry_times:
                wait_time = attempt * 5  # 递增等待时间
                print(f"⚠️  下载失败: {e}")
                print(f"   等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise OSError(f"下载失败（已重试 {retry_times} 次）: {e}") from e
                
        except Exception as e:
            last_error = e
            if attempt < retry_times:
                wait_time = attempt * 5
                print(f"⚠️  下载失败: {e}")
                print(f"   等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise OSError(f"下载失败（已重试 {retry_times} 次）: {e}") from e
    
    # 如果所有重试都失败
    raise OSError(f"下载失败（已重试 {retry_times} 次）: {last_error}") from last_error


def download_model_simple(
    repo_id: str,
    local_dir: str,
    max_workers: int = 16
) -> str:
    """
    简化版下载函数（使用默认配置）
    
    Args:
        repo_id: 模型仓库ID
        local_dir: 本地保存目录
        max_workers: 最大并发下载线程数
    
    Returns:
        本地模型目录路径
    """
    return download_model(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=max_workers,
        resume_download=True,
        check_before_download=True
    )


def _format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


if __name__ == "__main__":
    # 测试示例
    print("HuggingFace 模型下载工具")
    print("使用 download_model() 函数下载模型")
