#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# server.py — 单模型服务入口（自包含模板，每个模型文件夹独立一份）
#
#   - 监听：0.0.0.0:8080（环境变量 HOST / PORT 可覆盖，默认固定 8080）
#   - 接口：GET /health （loading=503 / ready=200 / error=500）
#           POST /predict（统一信封 {"inputs":..., "params":...}）
#   - 模型：同目录 adapter.py 中的 Adapter 类（本文件动态加载）
#   - 权重：优先加载本目录 weights/（用户预置），缺省回退 Hugging Face
#
# 启动：python server.py   （或直接执行 start.sh / start.bat）
# ============================================================================

import base64
import importlib
import io
import os
import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
DEVICE = os.environ.get("DEVICE", "cuda:0")
_STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    """模型加载失败（对应 /predict 503 MODEL_LOAD_FAILED）。"""


class InvalidInputError(ValueError):
    """输入校验失败（对应 /predict 400 INVALID_INPUT）。"""


class InferenceError(RuntimeError):
    """推理执行失败（对应 /predict 500 INFERENCE_ERROR）。"""


# ---------------------------------------------------------------------------
# 适配器状态机与基类
# ---------------------------------------------------------------------------
class AdapterState:
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class BaseAdapter(ABC):
    """所有模型适配器基类（与部署平台框架契约一致）。

    子类必须声明 MODELS 并实现 _load() / _predict()。
    第三方库一律在 _load() 内延迟导入。
    """

    category: ClassVar[str] = "common"
    MODELS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, model_id: str, device: str = DEVICE, config: dict | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.config = config or {}

        # 本地权重目录：本文件所在目录下的 weights/（用户预置权重）
        self._weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        if not os.path.isdir(self._weights_dir):
            self._weights_dir = None
        # 加载引用：weights/ 存在则加载本地，否则回退 HF 模型 ID
        self._load_ref = self._weights_dir if self._weights_dir else self.model_id

        self.state = AdapterState.IDLE
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_error: str | None = None
        self._loaded_at: float | None = None
        self._created_at = time.time()

    # 子类实现点
    @abstractmethod
    def _load(self) -> None:
        """延迟导入依赖并加载模型。"""

    @abstractmethod
    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """执行推理，返回可 JSON 序列化的 dict。"""

    # 模板方法（一般无需覆写）
    def load(self) -> None:
        with self._load_lock:
            if self.state == AdapterState.READY:
                return
            self.state = AdapterState.LOADING
            self._load_error = None
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - 需捕获全部异常转为 ERROR 状态
                self.state = AdapterState.ERROR
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise ModelLoadError(self._load_error) from exc
            self.state = AdapterState.READY
            self._loaded_at = time.time()

    def predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        self.load()
        with self._infer_lock:
            try:
                return self._predict(inputs, params)
            except (ModelLoadError, InvalidInputError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "category": self.category,
            "state": self.state,
            "model_ready": self.state == AdapterState.READY,
            "load_error": self._load_error,
            "loaded_at": self._loaded_at,
            "uptime_s": round(time.time() - self._created_at, 1),
            "device": self.device,
        }


# ---------------------------------------------------------------------------
# base64 编解码工具
# ---------------------------------------------------------------------------
def b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_to_image(data: str) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def image_to_b64(image: Any, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return bytes_to_b64(buf.getvalue())


def b64_to_audio(data: str, sample_rate: int | None = None):
    import torchaudio

    buf = io.BytesIO(base64.b64decode(data))
    waveform, sr = torchaudio.load(buf)
    if sample_rate is not None and sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    return waveform, sr


def audio_to_b64(waveform: Any, sample_rate: int, fmt: str = "wav") -> str:
    import torchaudio

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format=fmt)
    return bytes_to_b64(buf.getvalue())


def numpy_audio_to_b64(audio_np: Any, sample_rate: int) -> str:
    import numpy as np
    import torch
    import torchaudio

    tensor = torch.from_numpy(np.asarray(audio_np)).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, tensor, sample_rate, format="wav")
    return bytes_to_b64(buf.getvalue())


# ---------------------------------------------------------------------------
# GPU 信息（nvidia-smi 子进程，避免 pynvml 依赖）
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "devices": []}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return {"available": False, "devices": []}
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        devices.append(
            {
                "id": int(parts[0]),
                "name": parts[1],
                "vram_total_gb": round(int(parts[2]) / 1024, 1),
                "vram_used_gb": round(int(parts[3]) / 1024, 1),
                "gpu_util_pct": int(parts[4]),
            }
        )
    return {"available": True, "devices": devices}


# ---------------------------------------------------------------------------
# FastAPI 服务：/health + /predict
# ---------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="AI Model Service", version="0.1.0")

# 动态加载同目录 adapter 模块（置于所有定义之后，避免循环导入）
_adapter_mod = importlib.import_module("adapter")
ADAPTER_CLASS = _adapter_mod.Adapter
MODEL_ID = ADAPTER_CLASS.MODELS[0]
adapter = ADAPTER_CLASS(model_id=MODEL_ID, device=DEVICE, config={})


class PredictRequest(BaseModel):
    inputs: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None  # 可选：显式指定模型


@app.get("/health")
def health() -> JSONResponse:
    body: dict[str, Any] = {
        "status": "error",
        "version": "0.1.0",
        "model": MODEL_ID,
        "category": ADAPTER_CLASS.category,
        "gpu": get_gpu_info(),
        "uptime_s": round(time.time() - _STARTED_AT, 1),
    }
    body.update(adapter.health())
    if adapter.state == AdapterState.READY:
        body["status"] = "ready"
        return JSONResponse(status_code=200, content=body)
    if adapter.state == AdapterState.ERROR:
        body["status"] = "error"
        return JSONResponse(status_code=500, content=body)
    body["status"] = "loading"
    return JSONResponse(status_code=503, content=body)


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        outputs = adapter.predict(req.inputs, req.params)
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail={"code": "MODEL_LOAD_FAILED", "message": str(exc)}) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": str(exc)}) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail={"code": "INFERENCE_ERROR", "message": str(exc)}) from exc
    except Exception as exc:  # noqa: BLE001 - 兜底
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"},
        ) from exc
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {"outputs": outputs, "latency_ms": latency_ms, "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    print(f"[server] {MODEL_ID} 启动于 http://{HOST}:{PORT}（/health /predict）")
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
