# 部署指南

AI 多媒体平台生产环境部署完整指南。

[English](DEPLOYMENT.md) | **中文**

## 目录

1. [部署选项](#部署选项)
2. [Docker 部署](#docker-部署)
3. [手动部署](#手动部署)
4. [云部署](#云部署)
5. [配置](#配置)
6. [安全](#安全)
7. [监控](#监控)
8. [扩展](#扩展)

---

## 部署选项

| 方式 | 适用场景 | 复杂度 |
|------|----------|--------|
| Docker Compose | 单服务器、开发环境 | 简单 |
| Kubernetes | 生产环境、弹性扩展 | 中等 |
| 手动部署 | 自定义配置 | 中等 |
| 云平台 | 托管基础设施 | 简单-中等 |

---

## Docker 部署

### 前置条件

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker 运行时（用于 GPU）

### 快速开始

```bash
# 克隆仓库
git clone <repository-url>
cd aipicture

# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  models:
  outputs:
```

### GPU 支持

确保已安装 NVIDIA Docker 运行时：

```bash
# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## 手动部署

### 后端配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r backend/requirements.txt

# 安装带 CUDA 支持的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 使用 Gunicorn 运行

```bash
# 安装 Gunicorn
pip install gunicorn uvicorn

# 运行后端
gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### 前端构建

```bash
cd vue-web

# 安装依赖
npm install

# 生产构建
npm run build

# 使用 nginx 部署
# 将 dist/ 复制到 nginx www 目录
```

### Nginx 配置

```nginx
# /etc/nginx/sites-available/ai-platform
server {
    listen 80;
    server_name your-domain.com;

    # 前端
    location / {
        root /var/www/ai-platform;
        try_files $uri $uri/ /index.html;
    }

    # 后端 API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Systemd 服务

```ini
# /etc/systemd/system/ai-platform.service
[Unit]
Description=AI 多媒体平台后端
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-platform
Environment=PATH=/opt/ai-platform/venv/bin
ExecStart=/opt/ai-platform/venv/bin/gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 云部署

### AWS EC2

**推荐实例：**
- GPU: g4dn.xlarge (T4 GPU, 16GB 显存)
- 存储: 100GB+ SSD
- AMI: Deep Learning AMI（预装 CUDA）

**配置步骤：**
```bash
# 连接实例
ssh -i key.pem ubuntu@your-instance-ip

# 克隆并配置
git clone <repository>
cd aipicture
./scripts/setup.sh

# 启动服务
docker-compose up -d
```

### Google Cloud

**推荐配置：**
- n1-standard-4 + T4 GPU
- Container-Optimized OS

### Azure

**推荐配置：**
- NC6s_v3 (V100 GPU)
- Azure Container Instances

---

## 配置

### 环境变量

```bash
# 后端
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
WORKERS=4

# 数据库（可选）
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Redis（任务队列）
REDIS_URL=redis://localhost:6379

# 模型
MODEL_CACHE_DIR=/opt/models
HF_HOME=/opt/huggingface

# GPU
CUDA_VISIBLE_DEVICES=0

# 安全
SECRET_KEY=your-secret-key
CORS_ORIGINS=["https://your-domain.com"]

# 限制
MAX_UPLOAD_SIZE=100MB
REQUEST_TIMEOUT=300
```

### 模型配置

编辑 `config/modules_config.py`：

```python
TEXT_TO_IMAGE_CONFIG = {
    "default_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "max_width": 1024,
    "max_height": 1024,
    "max_steps": 100
}
```

---

## 安全

### HTTPS 配置

```bash
# 使用 Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 认证（可选）

在 `backend/core/config.py` 中启用 API 认证：

```python
AUTH_ENABLED = True
AUTH_SECRET_KEY = os.getenv("SECRET_KEY")
```

### 速率限制

在 `backend/main.py` 中配置：

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/generate")
@limiter.limit("10/minute")
async def generate():
    pass
```

### 文件上传安全

- 验证文件类型
- 限制文件大小
- 恶意软件扫描（可选）
- 存储在隔离目录

---

## 监控

### 健康检查

```bash
# 后端健康状态
curl http://localhost:8000/health

# GPU 状态
curl http://localhost:8000/api/system/status
```

### 日志

在 `backend/core/config.py` 中配置日志：

```python
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "/var/log/ai-platform/app.log"
}
```

### 指标（Prometheus）

```python
# 添加到后端
from prometheus_client import Counter, Histogram

generation_counter = Counter('generations_total', '生成总数')
generation_latency = Histogram('generation_latency_seconds', '生成延迟')
```

---

## 扩展

### 水平扩展

高流量场景：

```yaml
# docker-compose.scale.yml
services:
  backend:
    deploy:
      replicas: 4
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
```

### 任务队列

使用 Celery 处理后台任务：

```python
# backend/tasks.py
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def generate_image(params):
    # 长时间运行的生成任务
    pass
```

### GPU 扩展

多 GPU 配置：

```bash
# 设置可见 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3

# 在 backend/services/model_service.py 中实现负载均衡
```

---

## 维护

### 备份

```bash
# 备份模型和输出
tar -czf backup-$(date +%Y%m%d).tar.gz models/ outputs/

# 备份到 S3
aws s3 sync models/ s3://your-bucket/models/
```

### 更新

```bash
# 拉取最新代码
git pull origin main

# 重新构建
docker-compose build
docker-compose up -d
```

### 模型更新

```bash
# 清除模型缓存
rm -rf models/transformers_cache/*

# 下次启动时重新下载
```

---

## 故障排除

### 容器无法启动

```bash
# 检查日志
docker-compose logs backend

# 检查 GPU 访问
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 性能缓慢

1. 检查 GPU 利用率：`nvidia-smi`
2. 检查内存使用：`free -h`
3. 检查磁盘 I/O：`iotop`

### 内存不足

1. 减小批次大小
2. 启用模型卸载
3. 使用较小的模型
4. 增加交换空间

---

## 生产环境检查清单

- [ ] 已启用 HTTPS
- [ ] 已设置环境变量
- [ ] 已配置防火墙
- [ ] 已计划备份
- [ ] 已启用监控
- [ ] 已配置日志
- [ ] 已启用速率限制
- [ ] 已激活健康检查
- [ ] 已验证 GPU
- [ ] 磁盘空间充足
