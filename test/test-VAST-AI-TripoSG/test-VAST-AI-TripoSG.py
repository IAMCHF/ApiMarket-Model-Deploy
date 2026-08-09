#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# VAST-AI/TripoSG 服务测试脚本（自动生成，可自行扩展测试数据）
#
# 用法：
#   本机验证:  python test-VAST-AI-TripoSG.py                  # 默认 localhost:8080
#   内网部署:  python test-VAST-AI-TripoSG.py --host 10.0.62.60 --port <映射端口>
#
# 测试数据预置：本目录 data/ 下（image.png / audio.wav / texts.json 等）
# 文件类输入统一 base64 传输。
# 验证 /health 与 /predict 两个接口；退出码 0=PASS 1=FAIL。
# ============================================================================

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_text(name: str) -> str:
    """读取 data/ 下的文本文件。"""
    with open(os.path.join(_DATA_DIR, name), encoding="utf-8") as f:
        return f.read().strip()


def _load_json(name: str):
    """读取 data/ 下的 JSON 文件。"""
    with open(os.path.join(_DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _load_b64(name: str) -> str:
    """读取 data/ 下的二进制文件并转 base64（文件类输入统一 base64 传输）。"""
    with open(os.path.join(_DATA_DIR, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


def gen_inputs():
    """构造 /predict 请求的 inputs 字段（数据来自本目录 data/）。"""
    return {'image': _load_b64('image.png')}


def main() -> int:
    parser = argparse.ArgumentParser(description="VAST-AI/TripoSG 服务测试")
    parser.add_argument("--host", default="localhost", help="服务地址（内网: 10.0.62.60）")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--timeout", type=int, default=900, help="预测超时（秒，含模型加载）")
    parser.add_argument("--params", default="{}", help="额外推理参数（JSON 字符串）")
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"
    ok = True

    # ---- 1. /health（初始状态：模型懒加载，predict 前可能 503 loading，仅记录） ----
    try:
        resp = urllib.request.urlopen(base + "/health", timeout=15)
        body = json.loads(resp.read().decode())
        print(f"[health] {resp.status} status={body.get('status')} model={body.get('model')}")
    except Exception as exc:
        print(f"[health] 连接失败: {exc}")
        return 1

    # ---- 2. /predict ----
    try:
        params = json.loads(args.params or "{}")
        payload = json.dumps({"inputs": gen_inputs(), "params": params}).encode()
        req = urllib.request.Request(
            base + "/predict", data=payload, headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=args.timeout)
        body = json.loads(resp.read().decode())
        outs = body.get("outputs", {})
        print(f"[predict] {resp.status} 成功 latency_ms={body.get('latency_ms')}")
        print(f"[predict] outputs 键: {list(outs.keys())}")
        for key, val in outs.items():
            if isinstance(val, (list, tuple)):
                head = str(val[0])[:80] if val else "(空)"
                print(f"          {key}: [{len(val)} 项] 首项={head}")
            elif isinstance(val, str) and len(val) > 80:
                print(f"          {key}: {val[:80]}...")
            else:
                print(f"          {key}: {val}")
    except urllib.error.HTTPError as exc:
        print(f"[predict] {exc.code} {exc.read().decode()[:300]}")
        ok = False
    except Exception as exc:
        print(f"[predict] 失败: {exc}")
        ok = False

    # ---- 3. /health（推理后确认模型就绪） ----
    try:
        resp = urllib.request.urlopen(base + "/health", timeout=15)
        body = json.loads(resp.read().decode())
        print(f"[health] 推理后 status={body.get('status')} model_ready={body.get('model_ready')}")
        if body.get("status") != "ready":
            print(f"[health] 模型未就绪: state={body.get('state')} error={body.get('load_error')}")
            ok = False
    except Exception as exc:
        print(f"[health] 推理后检查失败: {exc}")
        ok = False

    print("[结果] PASS" if ok else "[结果] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
