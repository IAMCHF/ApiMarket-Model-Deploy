"""
google/timesfm-2.5-200m-pytorch 专用适配器（自动生成，请勿手工编辑）。
权重加载：优先从本文件夹 weights/ 加载，缺省回退 Hugging Face。
输入输出契约与部署平台框架一致（见 server.py 与 weights/README.txt）。
"""

from typing import Any
from server import (
    BaseAdapter,
    ModelLoadError,
    InvalidInputError,
    InferenceError,
    b64_to_bytes,
    bytes_to_b64,
    b64_to_image,
    image_to_b64,
    b64_to_audio,
    audio_to_b64,
    numpy_audio_to_b64,
)


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

class TimesFMAdapter(BaseAdapter):
    """Google TimesFM 零样本时序预测（google/timesfm-2.5-200m-pytorch）。

    官方 timesfm 包的加载 / 推理 API 随版本演进，以下为 best-effort 实现；
    上线前需按部署镜像中的 timesfm 版本校准。
    """

    MODELS = ('google/timesfm-2.5-200m-pytorch',)

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

    category = 'other'



# 统一适配器入口（server.py 加载此名称）
Adapter = TimesFMAdapter
