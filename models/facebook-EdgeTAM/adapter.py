"""
facebook/EdgeTAM 专用适配器（自动生成，请勿手工编辑）。
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


class EdgeTAMAdapter(BaseAdapter):
    """facebook/EdgeTAM 实时视频分割（边缘设备部署）。

    官方推理代码待接入 — see TODO。
    待接入内容：EdgeTAM 官方 checkpoint 与推理脚本
    （单帧 prompt 交互分割 / 视频流帧间传播分割）。
    """

    MODELS = ('facebook/EdgeTAM',)

    def _load(self) -> None:
        raise NotImplementedError("facebook/EdgeTAM 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("facebook/EdgeTAM 官方推理代码待接入 — see TODO")

    category = 'detection'



# 统一适配器入口（server.py 加载此名称）
Adapter = EdgeTAMAdapter
