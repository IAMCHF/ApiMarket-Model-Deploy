# ============================================================================
# FastAPI 服务入口：GET /health + POST /predict
#
# 启动方式（由 entrypoint.sh 执行）：
#   python -m uvicorn app.server:app --host 0.0.0.0 --port ${SERVICE_PORT}
#
# /health 语义（区分 liveness 与 readiness）：
#   - status=loading : 模型加载中  -> HTTP 503
#   - status=ready   : 模型就绪    -> HTTP 200
#   - status=error   : 加载失败    -> HTTP 500
#
# /predict 统一信封：
#   POST /predict
#   { "inputs": {...}, "params": {...} }          # model 由容器环境变量决定
#   { "model": "...", "inputs": {...}, "params": {...} }  # 可选显式指定
#   响应: { "outputs": {...}, "latency_ms": 123, "model": "..." }
#   错误: { "detail": { "code": "...", "message": "..." } }
#     400 INVALID_INPUT | 503 MODEL_LOAD_FAILED | 500 INFERENCE_ERROR
# ============================================================================

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .adapters import create_adapter
from .adapters.base import (
    AdapterState,
    InferenceError,
    InvalidInputError,
    ModelLoadError,
)
from .config import RuntimeConfig
from .utils.gpu import get_gpu_info

app = FastAPI(title="AI Model Deploy Platform", version="0.1.0")

cfg = RuntimeConfig()

# 根据 MODEL_ID 构建适配器；MODEL_ID 缺失或未注册时服务仍可启动，
# 便于框架导入校验，但 /health 与 /predict 返回明确错误。
adapter = None
_ADAPTER_ERROR: str | None = None
if not cfg.model_id:
    _ADAPTER_ERROR = "MODEL_ID environment variable is not set"
else:
    try:
        adapter = create_adapter(cfg.model_id, device=cfg.device, config=cfg.model_config)
    except KeyError as exc:
        _ADAPTER_ERROR = str(exc)

_STARTED_AT = time.time()
_VERSION = "0.1.0"


class PredictRequest(BaseModel):
    inputs: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None  # 可选：显式指定模型（单容器多模型场景）


@app.get("/health")
def health() -> JSONResponse:
    body: dict[str, Any] = {
        "status": "error",
        "version": _VERSION,
        "model": cfg.model_id,
        "model_type": cfg.model_type,
        "gpu": get_gpu_info(),
        "uptime_s": round(time.time() - _STARTED_AT, 1),
    }

    if adapter is None:
        body["reason"] = _ADAPTER_ERROR or "adapter not initialized"
        return JSONResponse(status_code=500, content=body)

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
    if adapter is None:
        raise HTTPException(
            status_code=500,
            detail={"code": "NO_MODEL", "message": _ADAPTER_ERROR or "adapter not initialized"},
        )

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
    return {"outputs": outputs, "latency_ms": latency_ms, "model": cfg.model_id}
