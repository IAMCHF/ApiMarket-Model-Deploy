"""
openbmb/VoxCPM2 专用适配器（自动生成，请勿手工编辑）。
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


class VoxCPMAdapter(BaseAdapter):
    """OpenBMB VoxCPM2 语音合成（voxcpm 包）。

    官方仓库：https://github.com/OpenBMB/VoxCPM
    """

    MODELS = ('openbmb/VoxCPM2',)

    def _load(self) -> None:
        import torch
        # TODO: 按官方 README 校准加载方式（VoxCPMConfig + VoxCPM）
        from voxcpm import VoxCPM, VoxCPMConfig

        config = VoxCPMConfig.from_pretrained(self._load_ref, trust_remote_code=True)
        self._model = VoxCPM.from_pretrained(self._load_ref,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        )
        self._model.eval().to(self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        speaker_id = int(speaker) if str(speaker).isdigit() else 0

        with torch.inference_mode():
            # TODO: VoxCPM 使用 fixed_prompt / mm_token 等输入，按官方 README 校准
            output = self._model.generate(text=text, speaker_id=speaker_id, max_new_tokens=2048)

        if isinstance(output, tuple):
            waveform, sr = output
        else:
            waveform, sr = output, int(params.get("sample_rate", 16000))
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = VoxCPMAdapter
