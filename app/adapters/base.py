# ============================================================================
# 适配器抽象基类与状态机
#
# 设计约定（所有适配器必须遵守）：
#   1. 所有第三方库（torch / transformers / diffusers / paddle ...）必须在
#      _load() 内延迟导入，保证框架代码可在未安装专用依赖的环境中
#      被导入与静态校验。
#   2. 子类只需实现 _load() 与 _predict() 两个方法。
#   3. predict() 为线程安全模板方法：自动确保模型已加载，并串行化推理。
#   4. /predict 请求体中的 inputs / params 直接透传给 _predict()。
# ============================================================================

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class ModelLoadError(RuntimeError):
    """模型加载失败（对应 /predict 503 MODEL_LOAD_FAILED）。"""


class InvalidInputError(ValueError):
    """输入校验失败（对应 /predict 400 INVALID_INPUT）。"""


class InferenceError(RuntimeError):
    """推理执行失败（对应 /predict 500 INFERENCE_ERROR）。"""


class AdapterState:
    """适配器生命周期状态。"""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class BaseAdapter(ABC):
    """所有模型适配器的基类。

    子类必须声明类属性 category 与 MODELS，并实现 _load() / _predict()。
    """

    # 类别 key，与 /deploy/requirements/{category}.txt 对应
    category: ClassVar[str] = "common"
    # 本适配器覆盖的模型 ID 列表；空元组 () 表示通配（不推荐，需最后匹配）
    MODELS: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.config = config or {}

        self.state = AdapterState.IDLE
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_error: str | None = None
        self._loaded_at: float | None = None
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # 子类实现点
    # ------------------------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """延迟导入依赖并加载模型（权重放持久卷 /data/models）。"""

    @abstractmethod
    def _predict(
        self, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """执行推理，返回可 JSON 序列化的 dict。"""

    # ------------------------------------------------------------------
    # 模板方法（一般无需覆写）
    # ------------------------------------------------------------------
    def load(self) -> None:
        """确保模型已加载（幂等，线程安全）。失败置 ERROR 状态并抛 ModelLoadError。"""
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

    def predict(
        self, inputs: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """线程安全推理入口：先确保加载，再串行化推理。"""
        self.load()
        with self._infer_lock:
            try:
                return self._predict(inputs, params)
            except (ModelLoadError, InvalidInputError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(f"{type(exc).__name__}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        """适配器健康信息（由 /health 聚合返回）。"""
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
