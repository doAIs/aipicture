# Deployment Guide

Complete guide for deploying the AI Multimedia Platform in production.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Docker Deployment](#docker-deployment)
3. [Manual Deployment](#manual-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration](#configuration)
6. [Security](#security)
7. [Monitoring](#monitoring)
8. [Scaling](#scaling)

---

## Deployment Options

| Method | Best For | Complexity |
|--------|----------|------------|
| Docker Compose | Single server, development | Easy |
| Kubernetes | Production, scaling | Medium |
| Manual | Custom setups | Medium |
| Cloud Platforms | Managed infrastructure | Easy-Medium |

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU)

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd aipicture

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f
```

### Docker Compose Configuration

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

### GPU Support

Ensure NVIDIA Docker runtime is installed:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Manual Deployment

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Running with Gunicorn

```bash
# Install Gunicorn
pip install gunicorn uvicorn

# Run backend
gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Frontend Build

```bash
cd vue-web

# Install dependencies
npm install

# Build for production
npm run build

# Serve with nginx
# Copy dist/ to nginx www directory
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/ai-platform
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/ai-platform;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
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

### Systemd Service

```ini
# /etc/systemd/system/ai-platform.service
[Unit]
Description=AI Multimedia Platform Backend
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

## Cloud Deployment

### AWS EC2

**Recommended Instance:**
- GPU: g4dn.xlarge (T4 GPU, 16GB VRAM)
- Storage: 100GB+ SSD
- AMI: Deep Learning AMI (CUDA pre-installed)

**Setup:**
```bash
# Connect to instance
ssh -i key.pem ubuntu@your-instance-ip

# Clone and setup
git clone <repository>
cd aipicture
./scripts/setup.sh

# Start services
docker-compose up -d
```

### Google Cloud

**Recommended:**
- n1-standard-4 + T4 GPU
- Container-Optimized OS

### Azure

**Recommended:**
- NC6s_v3 (V100 GPU)
- Azure Container Instances

---

## Configuration

### Environment Variables

```bash
# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
WORKERS=4

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Redis (for task queue)
REDIS_URL=redis://localhost:6379

# Models
MODEL_CACHE_DIR=/opt/models
HF_HOME=/opt/huggingface

# GPU
CUDA_VISIBLE_DEVICES=0

# Security
SECRET_KEY=your-secret-key
CORS_ORIGINS=["https://your-domain.com"]

# Limits
MAX_UPLOAD_SIZE=100MB
REQUEST_TIMEOUT=300
```

### Model Configuration

Edit `config/modules_config.py`:

```python
TEXT_TO_IMAGE_CONFIG = {
    "default_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "max_width": 1024,
    "max_height": 1024,
    "max_steps": 100
}
```

---

## Security

### HTTPS Setup

```bash
# Using Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Authentication (Optional)

Enable API authentication in `backend/core/config.py`:

```python
AUTH_ENABLED = True
AUTH_SECRET_KEY = os.getenv("SECRET_KEY")
```

### Rate Limiting

Configure in `backend/main.py`:

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.get("/api/generate")
@limiter.limit("10/minute")
async def generate():
    pass
```

### File Upload Security

- Validate file types
- Limit file sizes
- Scan for malware (optional)
- Store in isolated directory

---

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# GPU status
curl http://localhost:8000/api/system/status
```

### Logging

Configure logging in `backend/core/config.py`:

```python
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "/var/log/ai-platform/app.log"
}
```

### Metrics (Prometheus)

```python
# Add to backend
from prometheus_client import Counter, Histogram

generation_counter = Counter('generations_total', 'Total generations')
generation_latency = Histogram('generation_latency_seconds', 'Generation latency')
```

---

## Scaling

### Horizontal Scaling

For high traffic:

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

### Task Queue

Use Celery for background tasks:

```python
# backend/tasks.py
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def generate_image(params):
    # Long-running generation
    pass
```

### GPU Scaling

Multiple GPUs:

```bash
# Set visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3

# Load balance across GPUs
# Implement in backend/services/model_service.py
```

---

## Maintenance

### Backup

```bash
# Backup models and outputs
tar -czf backup-$(date +%Y%m%d).tar.gz models/ outputs/

# Backup to S3
aws s3 sync models/ s3://your-bucket/models/
```

### Updates

```bash
# Pull latest
git pull origin main

# Rebuild
docker-compose build
docker-compose up -d
```

### Model Updates

```bash
# Clear model cache
rm -rf models/transformers_cache/*

# Re-download on next start
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs backend

# Check GPU access
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### Slow Performance

1. Check GPU utilization: `nvidia-smi`
2. Check memory usage: `free -h`
3. Check disk I/O: `iotop`

### Out of Memory

1. Reduce batch size
2. Enable model offloading
3. Use smaller models
4. Add swap space

---

## Production Checklist

- [ ] HTTPS enabled
- [ ] Environment variables set
- [ ] Firewall configured
- [ ] Backups scheduled
- [ ] Monitoring enabled
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] Health checks active
- [ ] GPU verified
- [ ] Disk space adequate
