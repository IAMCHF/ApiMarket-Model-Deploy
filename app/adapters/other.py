# ============================================================================
# 其他专项模型适配器（7 个模型）
#
# 四个适配器族：
#   1. ChronosAdapter      —— Amazon Chronos 时序预测（chronos-2 / chronos-bolt 系列）
#   2. TimesFMAdapter      —— Google TimesFM 零样本时序预测
#   3. PrivacyFilterAdapter—— openai/privacy-filter 隐私 / 敏感信息文本分类
#   4. AntaresAdapter      —— Cisco Antares-1B 代码漏洞定位 / 安全检测
#
# /predict 输入输出约定：
#   时序预测：{"context": [float, ...], "prediction_length": int}
#          -> {"forecast": [...], "mean": float, "quantiles": {"0.1": [...], ...}}
#   隐私过滤：{"text": "..."} -> {"flagged": bool, "score": float, "label": str}
# ============================================================================

from typing import Any

from .base import BaseAdapter, InvalidInputError, ModelLoadError
from .registry import register_adapter


# ---------------------------------------------------------------------------
# 时序预测公共校验：context / prediction_length
# ---------------------------------------------------------------------------
def _validate_forecast_inputs(inputs: dict[str, Any]) -> tuple[list[float], int]:
    """校验时序预测输入，返回 (context, prediction_length)。"""
    context = inputs.get("context")
    if not isinstance(context, (list, tuple)) or len(context) == 0:
        raise InvalidInputError("inputs 必须包含非空的 'context'（float 数值列表）")
    if not all(isinstance(x, (int, float)) for x in context):
        raise InvalidInputError("'context' 中的所有元素必须为数值")

    prediction_length = inputs.get("prediction_length")
    if not isinstance(prediction_length, int) or prediction_length <= 0:
        raise InvalidInputError("inputs 必须包含正整数 'prediction_length'")

    return [float(x) for x in context], prediction_length


@register_adapter(category="other")
class ChronosAdapter(BaseAdapter):
    """Amazon Chronos 时序预测（chronos-2 / chronos-bolt 系列）。

    bolt 系列（base/large/mini）同为 T5 架构，统一经
    ChronosPipeline.from_pretrained 加载。预测采样多条轨迹后返回
    逐时间步均值与 [0.1, 0.5, 0.9] 分位数。
    """

    MODELS = (
        "amazon/chronos-2",
        "amazon/chronos-bolt-base",
        "amazon/chronos-bolt-large",
        "amazon/chronos-bolt-mini",
    )

    # 不同规模的默认采样数（param num_samples 可覆盖）
    _DEFAULT_NUM_SAMPLES = {
        "amazon/chronos-2": 20,
        "amazon/chronos-bolt-large": 20,
        "amazon/chronos-bolt-base": 64,
        "amazon/chronos-bolt-mini": 64,
    }

    def _load(self) -> None:
        """延迟导入 chronos 官方包，加载 ChronosPipeline。"""
        try:
            import torch
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：chronos-forecasting 未安装，无法加载 Chronos 模型（{exc}）"
            ) from exc

        self._device_obj = torch.device(self.device)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=self._device_obj.type if self._device_obj.type != "cpu" else "cpu",
            torch_dtype=(
                torch.bfloat16 if self._device_obj.type == "cuda" else torch.float32
            ),
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        context, prediction_length = _validate_forecast_inputs(inputs)
        num_samples = int(
            params.get("num_samples", self._DEFAULT_NUM_SAMPLES.get(self.model_id, 20))
        )
        if num_samples <= 0:
            raise InvalidInputError("'num_samples' 必须为正整数")

        # 采样多条轨迹，shape = [num_samples, prediction_length]
        forecast = self._pipeline.predict(
            context=torch.tensor(context, dtype=torch.float32, device=self._device_obj),
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        mean = forecast.mean(dim=0)  # [prediction_length]
        q = torch.quantile(forecast, torch.tensor([0.1, 0.5, 0.9]), dim=0)  # [3, prediction_length]

        return {
            "forecast": mean.tolist(),
            "mean": float(mean.mean().item()),
            "quantiles": {
                "0.1": q[0].tolist(),
                "0.5": q[1].tolist(),
                "0.9": q[2].tolist(),
            },
        }


@register_adapter(category="other")
class TimesFMAdapter(BaseAdapter):
    """Google TimesFM 零样本时序预测（google/timesfm-2.5-200m-pytorch）。

    官方 timesfm 包的加载 / 推理 API 随版本演进，以下为 best-effort 实现；
    上线前需按部署镜像中的 timesfm 版本校准。
    """

    MODELS = (
        "google/timesfm-2.5-200m-pytorch",
    )

    def _load(self) -> None:
        """延迟导入 timesfm 官方包并加载 checkpoint。

        TODO(上线前必做)：按官方 README 校准
          1. TimesFmHparams 的字段（num_layers / model_dims / backend 等）是否
             随 timesfm-2.5 有变化；
          2. TimesFmCheckpoint 的 repo id 与本地权重路径映射
             （权重应由 tools/download_models.py 拉取到 $MODELS_ROOT）。
        """
        try:
            import timesfm
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：timesfm 官方包未安装，无法加载 TimesFM 模型（{exc}）"
            ) from exc

        # 构造 hparams：backend 按当前设备选择 cpu / gpu
        backend = "gpu" if "cuda" in self.device else "cpu"
        hparams = timesfm.TimesFmHparams(
            per_core_batch_size=int(self.config.get("per_core_batch_size", 32)),
            horizon_len=int(self.config.get("horizon_len", 128)),
            num_layers=int(self.config.get("num_layers", 20)),
            model_dims=int(self.config.get("model_dims", 1280)),
            backend=backend,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.model_id,
        )
        try:
            self._tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            self._tfm.load_from_checkpoint()
        except Exception as exc:  # noqa: BLE001 - API 未校准，转骨架错误
            raise NotImplementedError(
                f"需接入 TimesFM 官方运行时 — see TODO（加载路径未校准：{exc}）"
            ) from exc

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import pandas as pd

        context, prediction_length = _validate_forecast_inputs(inputs)
        freq = str(params.get("freq", self.config.get("freq", "H")))

        # TimesFM 官方 API 以 DataFrame（unique_id / ds / y）为输入
        df = pd.DataFrame(
            {
                "unique_id": ["ts0"] * len(context),
                "ds": pd.date_range(end="2024-01-01", periods=len(context), freq=freq),
                "y": np.asarray(context, dtype=np.float64),
            }
        )

        # TODO：timesfm-2.5 返回结构（point_forecast / quantile_forecast 形状与
        # 分位点定义）需按版本校准。官方默认分位点为
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]。
        point_forecast, quantile_forecast = self._tfm.forecast(
            df=df,
            forecast_context_length=len(context),
            window_size=int(self.config.get("window_size", 512)),
        )
        # point_forecast: [num_ts, horizon]
        mean = [float(x) for x in point_forecast[0]]

        # quantile_forecast: [num_ts, num_quantiles, horizon]，取 0.1/0.5/0.9
        quantiles: dict[str, list[float]] = {}
        q_idx = {"0.1": 0, "0.5": 4, "0.9": 8}
        for label, idx in q_idx.items():
            quantiles[label] = [
                float(x) for x in quantile_forecast[0, idx, :prediction_length]
            ]

        return {
            "forecast": mean[:prediction_length],
            "mean": float(np.mean(mean[:prediction_length]).item()),
            "quantiles": quantiles,
        }


@register_adapter(category="other")
class PrivacyFilterAdapter(BaseAdapter):
    """openai/privacy-filter 隐私 / 敏感信息文本过滤。

    该模型为文本序列分类器，输出 "flagged"（命中隐私/敏感信息）
    与 "safe" 两类标签。
    """

    MODELS = (
        "openai/privacy-filter",
    )

    def _load(self) -> None:
        """延迟导入 transformers，加载序列分类模型。"""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：transformers / torch 未安装，无法加载隐私过滤器（{exc}）"
            ) from exc

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.nn.functional import softmax

        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("inputs 必须包含非空的 'text'（字符串）")

        encoded = self._tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device_obj)

        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probs = softmax(logits, dim=-1)

        max_idx = int(torch.argmax(logits, dim=-1).item())
        score = float(probs[0, max_idx].item())
        # 优先使用模型 id2label，兜底按 0/1 位置映射
        id2label = getattr(self._model.config, "id2label", None) or {}
        label = str(id2label.get(max_idx, "flagged" if max_idx == 1 else "safe"))
        flagged = "flagged" in label.lower()

        return {"flagged": flagged, "score": score, "label": label}


@register_adapter(category="other")
class AntaresAdapter(BaseAdapter):
    """Cisco Antares-1B 代码漏洞定位 / 安全检测。

    注意：Antares 专为气隙（air-gapped）安全审计场景设计，部署环境通常
    无法联网拉取权重，须预先在隔离网络内把模型权重镜像到 $MODELS_ROOT。
    当前为骨架实现 —— 真实推理需接入 Cisco Antares 官方推理运行时。
    """

    MODELS = (
        "cisco-ai/Antares-1B",
    )

    def _load(self) -> None:
        """延迟导入 Cisco Antares 官方运行时并加载模型。

        TODO(上线前必做)：
          1. 确认 Antares 官方推理运行时（pip 包 / Docker 镜像）的导入路径，
             在气隙环境中以镜像方式离线安装；
          2. 权重由气隙环境内预置到 $MODELS_ROOT/cisco-ai/Antares-1B；
          3. 校准漏洞定位的输入格式（源码片段 / 文件路径）与输出结构
             （漏洞位置、严重度、CWE 类别等）。
        """
        raise NotImplementedError("需接入 Cisco Antares 运行时 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        # 骨架占位：仅做输入校验，真实推理见 _load 中的 TODO
        _ = inputs, params
        raise NotImplementedError("需接入 Cisco Antares 运行时 — see TODO")
