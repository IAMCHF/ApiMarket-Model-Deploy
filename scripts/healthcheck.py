#!/usr/bin/env python3
# ============================================================================
# healthcheck.py — 容器健康探针
#
# 用法：python healthcheck.py [--port 8000] [--timeout 5]
# 退出码：0=ready / 1=loading 或不可达（供 docker healthcheck / k8s 探针使用）
# ============================================================================

import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="容器健康探针")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=os_env_port())
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/health"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - 不可达/解析失败视为不健康
        return 1

    # 仅 status=ready 视为健康（loading 返回 503 会被 urlopen 抛异常，已在上方兜底）
    return 0 if body.get("status") == "ready" else 1


def os_env_port() -> str:
    import os

    return os.environ.get("SERVICE_PORT", "8000")


if __name__ == "__main__":
    sys.exit(main())
