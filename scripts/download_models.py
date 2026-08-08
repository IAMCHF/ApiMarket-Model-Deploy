# ============================================================================
# download_models.py — 按 MODEL_ID 预下载模型权重到持久卷
#
# 用法：
#   python download_models.py --model baidu/Unlimited-OCR [--models-config /deploy/config/models.yaml]
#
# 权重落地目录（环境变量）：MODELS_ROOT（默认 /data/models）
# 依赖 HF_ENDPOINT 支持国内镜像（如 https://hf-mirror.com）
# ============================================================================

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="下载模型权重到持久卷")
    parser.add_argument("--model", required=True, help="HF 模型 ID")
    parser.add_argument("--models-config", default=os.environ.get("MODELS_CONFIG", "/deploy/config/models.yaml"))
    args = parser.parse_args()

    models_root = Path(os.environ.get("MODELS_ROOT", "/data/models"))
    models_root.mkdir(parents=True, exist_ok=True)

    # 权重已存在则跳过（huggingface_hub 本身幂等，这里做快速短路）
    model_dir = models_root / "hf_cache" / "hub" / args.model.replace("/", "--")
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"[download] 权重已存在，跳过：{args.model} -> {model_dir}")
        return 0

    # 延迟导入：huggingface_hub 在镜像 L2 层已固化
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] huggingface_hub 不可用，请检查镜像 L2 层", file=sys.stderr)
        return 1

    print(f"[download] 开始下载：{args.model}")
    try:
        snapshot_download(repo_id=args.model, local_dir_use_symlinks=False)
    except Exception as exc:  # noqa: BLE001 - 下载失败不阻塞启动（可在线兜底）
        print(f"[download] 下载失败（服务启动后可能在线拉取）：{exc}", file=sys.stderr)
        return 1
    print(f"[download] 完成：{args.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
