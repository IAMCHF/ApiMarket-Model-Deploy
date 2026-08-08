#!/usr/bin/env bash
# ============================================================================
# 统一启动脚本 — 固定监听 0.0.0.0:8080
#
# 用法：./start.sh
# 前置：本机已安装 NVIDIA 驱动 + CUDA 运行时（容器/宿主机均可）
# 权重：请提前将模型权重放置于本目录 weights/ 下（见 weights/README.txt）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[start] 安装依赖..."
python -m pip install -r requirements.txt

echo "[start] 启动服务 http://0.0.0.0:8080 ..."
exec python server.py
