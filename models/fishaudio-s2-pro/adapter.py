"""
fishaudio/s2-pro 专用适配器（自动生成，请勿手工编辑）。
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


class FishSpeechAdapter(BaseAdapter):
    """fish-speech 语音合成（fishaudio/s2-pro）。

    官方仓库：https://github.com/fishaudio/fish-speech
    离线推理依赖该仓库的 config + checkpoint（.yaml/.ckpt），
    此处按官方推荐接入本地推理服务（OpenAI 兼容 API）。
    """

    MODELS = ('fishaudio/s2-pro',)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；s2-pro 仓库含 v1/v2 双版本，需按 config 指定
        from fish_speech.api import API  # 厂商客户端（在线/本地推理服务）

        self._api = API(
            model="s2-pro",
            base_url=self.config.get("api_base", "http://127.0.0.1:8000/v1"),
            api_key=self.config.get("api_key", ""),
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import base64

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        # TODO: 按官方 README 校准返回类型（bytes / PlaybackResponse / numpy）
        response = self._api.tts(text, voice=speaker or "default", format="wav")
        if hasattr(response, "audio"):
            audio_bytes = response.audio
        else:
            audio_bytes = bytes(response)
        sr = int(params.get("sample_rate", 44100))  # fish-speech 默认 44100
        return {
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": sr,
            "format": "wav",
        }

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = FishSpeechAdapter
