"""
amazon/chronos-bolt-mini 专用适配器（自动生成，请勿手工编辑）。
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

class ChronosAdapter(BaseAdapter):
    """Amazon Chronos 时序预测（chronos-2 / chronos-bolt 系列）。

    bolt 系列（base/large/mini）同为 T5 架构，统一经
    ChronosPipeline.from_pretrained 加载。预测采样多条轨迹后返回
    逐时间步均值与 [0.1, 0.5, 0.9] 分位数。
    """

    MODELS = ('amazon/chronos-bolt-mini',)

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
        self._pipeline = ChronosPipeline.from_pretrained(self._load_ref,
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

    category = 'other'



# 统一适配器入口（server.py 加载此名称）
Adapter = ChronosAdapter
