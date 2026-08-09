"""
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 专用适配器（自动生成，请勿手工编辑）。
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


class TransformersTTSAdapter(BaseAdapter):
    """transformers 原生 TTS：AutoModel 加载 Qwen3-TTS（qwen3_tts 架构）。

    支持通过 params.speaker_description 传入音色描述（CustomVoice 特性）。
    注：需要 transformers>=5.15（per-model 依赖覆盖，2026-08 实测 4.57/5.14 均不支持）。
    """

    MODELS = ('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref)
        self._model = AutoModel.from_pretrained(
            self._load_ref,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        tokens = self._tokenizer(text, return_tensors="pt").to(self._device_obj)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.get("max_new_tokens", self.config.get("max_new_tokens", 2048))),
        }
        if speaker:
            # 音色描述（CustomVoice 支持自然语言描述说话人音色）
            gen_kwargs["speaker_description"] = speaker

        with torch.inference_mode():
            output = self._model.generate(**tokens, **gen_kwargs)

        waveform = output.audio.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = int(getattr(output, "sample_rate", 24000))
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = TransformersTTSAdapter
