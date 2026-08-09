#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# CohereLabs/cohere-transcribe-03-2026 服务测试脚本（自动生成，可自行扩展测试数据）
#
# 用法：
#   本机验证:  python cohere-transcribe-03-2026.py                    # 默认 localhost:8080
#   内网部署:  python cohere-transcribe-03-2026.py --host 10.0.62.60 --port <映射端口>
#
# 验证 /health 与 /predict 两个接口；退出码 0=PASS 1=FAIL。
# ============================================================================

import argparse
import base64
import io
import json
import math
import struct
import sys
import urllib.error
import urllib.request


def _gen_image(text: str = "TEST 123", width: int = 640, height: int = 180) -> str:
    """生成一张含文字的图片（base64），供图像类模型测试。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = None
    ImageDraw.Draw(img).text((30, 60), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _gen_audio(duration: float = 1.5, freq: int = 440, sample_rate: int = 16000) -> str:
    """生成一段正弦波音频（base64 wav），供语音类模型测试。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * t / sample_rate)))
            for t in range(int(duration * sample_rate))
        )
        w.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


def gen_inputs():
    """生成该模型可用的测试输入（inputs 字段）。"""
    return {'audio': _gen_audio()}


def main() -> int:
    parser = argparse.ArgumentParser(description="CohereLabs/cohere-transcribe-03-2026 服务测试")
    parser.add_argument("--host", default="localhost", help="服务地址（内网: 10.0.62.60）")
    parser.add_argument("--port", type=int, default=8080, help="服务端口")
    parser.add_argument("--timeout", type=int, default=900, help="预测超时（秒，含模型加载）")
    parser.add_argument("--params", default="{}", help="额外推理参数（JSON 字符串）")
    args = parser.parse_args()
    base = f"http://{args.host}:{args.port}"
    ok = True

    # ---- 1. /health ----
    try:
        resp = urllib.request.urlopen(base + "/health", timeout=15)
        body = json.loads(resp.read().decode())
        print(f"[health] {resp.status} status={body.get('status')} model={body.get('model')}")
        if body.get("status") != "ready":
            print(f"[health] 模型未就绪: state={body.get('state')} error={body.get('load_error')}")
            ok = False
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        print(f"[health] {exc.code} status={body.get('status')}")
        ok = False
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

    print("[结果] PASS" if ok else "[结果] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
