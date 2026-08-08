"""
nvidia/instant-nurec 专用适配器（自动生成，请勿手工编辑）。
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


class InstantNuRecAdapter(BaseAdapter):
    """nvidia/instant-nurec 神经场景重建。

    官方推理代码待接入 — see TODO。
    待接入内容：instant-nurec 官方重建管线（多视角图像输入、
    体素/网格导出流程按官方 README 校准）。
    """

    MODELS = ('nvidia/instant-nurec',)

    def _load(self) -> None:
        raise NotImplementedError("nvidia/instant-nurec 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("nvidia/instant-nurec 官方推理代码待接入 — see TODO")

    category = '3d-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = InstantNuRecAdapter
