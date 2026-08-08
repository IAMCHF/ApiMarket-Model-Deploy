"""
canopylabs/orpheus-3b-0.1-ft 专用适配器（自动生成，请勿手工编辑）。
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


class OrpheusAdapter(BaseAdapter):
    """Canopy Labs Orpheus 语音合成（orpheus-tts 包）。

    官方仓库：https://github.com/canopyai/Orpheus-TTS
    """

    MODELS = ('canopylabs/orpheus-3b-0.1-ft',)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；orpheus-tts 提供 Orpheus 推理类
        from orpheus_tts import Orpheus

        self._model = Orpheus(model_path=self._load_ref, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        emotion = params.get("emotion", "None")  # Orpheus 支持 tara / leah 等情绪

        # TODO: Orpheus 先产出音素 token 序列，再经 tokenizer 解码为音频，按官方 README 校准
        syn_tokens = self._model.generate_speech(text, emotion=emotion, voice=speaker or "tara")
        waveform = self._model.tokenizer.decode(syn_tokens[0])
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = 24000
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = OrpheusAdapter
