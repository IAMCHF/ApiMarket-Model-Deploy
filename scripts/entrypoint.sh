#!/usr/bin/env bash
# ============================================================================
# entrypoint.sh — 容器启动入口（挂载自宿主机 /deploy/scripts）
#
# 流程：
#   1. 安装通用依赖 + 本模型类别依赖（命中 pip-cache 卷则秒级完成）
#   2. 预下载模型权重到 /data/models（已存在则跳过）
#   3. 启动 uvicorn 拉起 FastAPI 服务（/health + /predict）
# ============================================================================
set -euo pipefail

# --- 必需环境变量 ---
: "${MODEL_TYPE:?MODEL_TYPE 环境变量必须设置（如 ocr/tts/image-gen）}"
: "${MODEL_ID:?MODEL_ID 环境变量必须设置（如 baidu/Unlimited-OCR）}"

DEPLOY_ROOT="${DEPLOY_ROOT:-/deploy}"
REQUIREMENTS_DIR="${REQUIREMENTS_DIR:-$DEPLOY_ROOT/requirements}"
MODELS_CONFIG="${MODELS_CONFIG:-$DEPLOY_ROOT/config/models.yaml}"
SERVICE_PORT="${SERVICE_PORT:-8000}"

# --- HF 国内镜像（可选） ---
if [ -n "${HF_ENDPOINT:-}" ]; then
    export HF_ENDPOINT
fi

echo "[entrypoint] 安装依赖：common + ${MODEL_TYPE}"
pip install --no-cache-dir \
    -r "${REQUIREMENTS_DIR}/common.txt" \
    -r "${REQUIREMENTS_DIR}/${MODEL_TYPE}.txt"

echo "[entrypoint] 预下载模型权重：${MODEL_ID}"
python "${DEPLOY_ROOT}/scripts/download_models.py" \
    --model "${MODEL_ID}" \
    --models-config "${MODELS_CONFIG}"

echo "[entrypoint] 启动服务：${MODEL_ID} @ :${SERVICE_PORT}"
export PYTHONPATH="${DEPLOY_ROOT}"
exec python -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port "${SERVICE_PORT}" \
    --workers 1
