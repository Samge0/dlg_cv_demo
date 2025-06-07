# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY server.py .

# 创建缓存目录
RUN mkdir -p output

# 暴露端口
EXPOSE 9720

# 挂载缓存目录
VOLUME ["/app/output", "/root/.cache"]

# 启动命令
CMD ["python3", "server.py"] 