"""
sesame/csm-1b 专用适配器（自动生成，请勿手工编辑）。
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


class CSMAdapter(BaseAdapter):
    """Sesame CSM-1B 语音合成（sesame_csm 包）。

    官方仓库：https://github.com/SesameAILabs/csm
    """

    MODELS = ('sesame/csm-1b',)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；load_csm_1b 直接加载原生权重
        from sesame_csm.generator import load_csm_1b

        self._generator = load_csm_1b(self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        try:
            speaker_id = int(speaker) if speaker else 0
        except (TypeError, ValueError):
            speaker_id = 0

        with torch.inference_mode():
            # TODO: CSM 支持 context（多轮对话）输入，按官方 README 校准
            audio = self._generator.generate(
                text=text, speaker=speaker_id, max_audio_length_ms=int(params.get("max_audio_length_ms", 30000))
            )

        # 返回 numpy int16/float32 波形 -> 转 torch.Tensor 后编码
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        sr = 24000
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = CSMAdapter
