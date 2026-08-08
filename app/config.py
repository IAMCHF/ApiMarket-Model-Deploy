# ============================================================================
# 运行时配置：环境变量 + config/models.yaml（挂载自宿主机 /deploy/config）
# ============================================================================

import os
from typing import Any

import yaml

_MODELS_CONFIG_CACHE: dict[str, dict[str, Any]] | None = None


def load_models_config() -> dict[str, dict[str, Any]]:
    """加载 models.yaml，返回 {model_id: {...}}；文件缺失/异常时返回空 dict。"""
    global _MODELS_CONFIG_CACHE
    if _MODELS_CONFIG_CACHE is not None:
        return _MODELS_CONFIG_CACHE

    path = os.environ.get("MODELS_CONFIG", "/deploy/config/models.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _MODELS_CONFIG_CACHE = data.get("models", {}) or {}
    except FileNotFoundError:
        _MODELS_CONFIG_CACHE = {}
    except Exception:  # noqa: BLE001 - 配置解析失败不阻塞启动
        _MODELS_CONFIG_CACHE = {}
    return _MODELS_CONFIG_CACHE


class RuntimeConfig:
    """由容器运行时注入的环境变量构成的配置对象。"""

    def __init__(self) -> None:
        self.model_id: str = os.environ.get("MODEL_ID", "")
        self.model_type: str = os.environ.get("MODEL_TYPE", "")
        self.service_port: int = int(os.environ.get("SERVICE_PORT", "8000"))
        self.device: str = os.environ.get("DEVICE", "cuda:0")
        self.hf_endpoint: str = os.environ.get("HF_ENDPOINT", "")
        self.models_root: str = os.environ.get("MODELS_ROOT", "/data/models")
        self.outputs_root: str = os.environ.get("OUTPUTS_ROOT", "/data/outputs")
        # models.yaml 中该模型的条目（类别/显存/默认参数）
        self.model_config: dict[str, Any] = load_models_config().get(
            self.model_id, {}
        )
