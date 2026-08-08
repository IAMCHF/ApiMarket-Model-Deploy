"""
Zyphra/Zonos 专用适配器（自动生成，请勿手工编辑）。
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


class ZonosAdapter(BaseAdapter):
    """Zyphra Zonos 语音合成（ZonosPipeline）。

    官方仓库：https://github.com/Zyphra/Zonos
    注：Zonos 需要 speaker embedding / emotion 等条件输入，按 README 校准。
    """

    MODELS = ('Zyphra/Zonos',)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；优先 ZonosPipeline，回退到 Zonos 类
        try:
            from zonos import ZonosPipeline

            self._model = ZonosPipeline.from_pretrained(self._load_ref, device=self.device)
        except ImportError:
            from zonos import Zonos

            self._model = Zonos.from_pretrained(self._load_ref, device=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        gen_kwargs: dict[str, Any] = {}
        if speaker:
            # Zonos 支持 speaker description 作为条件输入
            gen_kwargs["speaker_description"] = speaker

        with torch.inference_mode():
            # TODO: synthesize 方法名与返回结构按官方 README 校准
            output = self._model.synthesize(text, **gen_kwargs)

        if isinstance(output, tuple):
            waveform, sr = output
        else:
            waveform, sr = output, int(params.get("sample_rate", 24000))
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = ZonosAdapter
