#!/usr/bin/env bash
# ============================================================================
# 统一启动脚本 — 固定监听 0.0.0.0:8080
#
# 用法：./start.sh
# 前置：本机已安装 NVIDIA 驱动 + CUDA 运行时（容器/宿主机均可）
# 第一步：换源（国内 pip 加速，不改镜像构建源；可用 PIP_INDEX_URL 环境变量覆盖）
# 权重：请提前将模型权重放置于本目录 weights/ 下（见 weights/README.txt）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- 1. 换源（国内 pip 加速；内网环境可设置 PIP_INDEX_URL 指向内网镜像） ----
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- 2. 安装依赖（命中 /data/pip-cache 卷则秒级完成） ----
echo "[start] 安装依赖（源: ${PIP_INDEX_URL}）..."
python -m pip install -r requirements.txt

# ---- 3. 启动服务（固定 0.0.0.0:8080） ----
echo "[start] 启动服务 http://0.0.0.0:8080 ..."
exec python server.py
