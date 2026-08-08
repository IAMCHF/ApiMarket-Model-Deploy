"""
microsoft/VibeVoice-Realtime-0.5B 专用适配器（自动生成，请勿手工编辑）。
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


class _TODOTTSAdapter(BaseAdapter):
    """TTS 骨架适配器基类：依赖厂商官方运行时，尚未接入。"""

    #: 厂商/运行时名称，用于异常信息
    _vendor = "unknown"

    def _load(self) -> None:
        raise NotImplementedError(
            f"requires official {self._vendor} runtime — see TODO（尚未接入 {self._vendor} 官方运行时）"
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f"requires official {self._vendor} runtime — see TODO（尚未接入 {self._vendor} 官方运行时）"
        )

class VibeVoiceAdapter(_TODOTTSAdapter):
    """microsoft/VibeVoice：微软实时语音合成运行时（骨架占位）。"""

    MODELS = ('microsoft/VibeVoice-Realtime-0.5B',)
    _vendor = "VibeVoice"

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = VibeVoiceAdapter
