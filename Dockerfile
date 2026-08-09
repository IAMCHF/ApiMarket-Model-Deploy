# ============================================================================
# AI 模型部署平台 - 基础镜像
# 基于 nvidia/cuda:12.1.1-cudnn8 + Python 3.11 + PyTorch 2.4 (cu121)
#
# 设计原则：
#   - 镜像只固化二进制运行环境（L1 系统层 + L2 框架层）
#   - 不含 requirements / 脚本 / 配置 / 模型权重（运行时挂载）
#   - 无 ENTRYPOINT / CMD，启动命令由外部注入
#   - 镜像总体积约 9GB
# ============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ===========================================================================
# L1 系统层：CUDA 12.1 + cuDNN + Python 3.11 + ffmpeg + tesseract + 中文字体
# ===========================================================================
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        # --- Python 3.11 ---
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        # --- 构建工具 ---
        build-essential \
        cmake \
        git \
        wget \
        curl \
        # --- 多媒体处理 ---
        ffmpeg \
        # --- OCR 引擎 ---
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        # --- 图像处理依赖（OpenCV 等） ---
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        # --- 音频处理依赖 ---
        libsndfile1 \
        # --- 中文字体 ---
        fonts-noto-cjk \
        # --- 其他 ---
        ca-certificates \
        tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.11 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

# ===========================================================================
# 国内 pip 镜像加速（可选；部署到纯内网后移除即可）
# ===========================================================================
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ===========================================================================
# L2 框架层：PyTorch 2.4 (cu121) + transformers + accelerate + 通用框架
# ===========================================================================

# 安装 PyTorch 2.4 + torchvision + torchaudio (cu121)
# 国内网络使用清华 pytorch-wheels 镜像加速；如镜像缺失可回退官方源：
#   --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu121

# 安装通用框架依赖（transformers / accelerate / bitsandbytes / onnxruntime 等）
COPY requirements-base.txt /tmp/requirements-base.txt
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt \
    && rm /tmp/requirements-base.txt

# ===========================================================================
# 环境变量（指向挂载路径，容器启动时由外部卷提供）
# ===========================================================================
ENV HF_HOME=/data/models/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/data/models/hf_cache/hub
ENV TRANSFORMERS_CACHE=/data/models/hf_cache/hub
ENV PIP_CACHE_DIR=/data/pip-cache
ENV TORCH_HOME=/data/models/torch_cache
ENV MODELS_ROOT=/data/models
ENV OUTPUTS_ROOT=/data/outputs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ===========================================================================
# 无 ENTRYPOINT / CMD — 启动命令完全由外部注入（docker run / compose）
# ===========================================================================
