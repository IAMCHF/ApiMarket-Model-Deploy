"""
OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B 专用适配器（自动生成，请勿手工编辑）。
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


class _AudioLLMAsrMixin:
    """transformers 音频大模型 ASR 共用推理流程：audio -> processor -> generate -> decode。"""

    def _transcribe_audio(self, inputs: dict[str, Any], params: dict[str, Any]) -> str:
        import torch

        audio_b64 = inputs.get("audio")
        if not audio_b64:
            raise InvalidInputError("inputs 必须包含 'audio'（base64 音频字符串）")

        waveform, sr = b64_to_audio(audio_b64)
        audio_array = waveform[0].numpy()  # [T] 单声道，取值范围 [-1, 1]

        batch = self._processor(
            audio=audio_array, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
        ).to(self._device_obj)

        max_new_tokens = int(params.get("max_new_tokens", self.config.get("max_new_tokens", 2048)))
        with torch.inference_mode():
            generated = self._model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)

        # 部分模型只返回 input_features、无 input_ids，此时整段解码（跳过特殊 token）
        prompt_len = batch["input_ids"].shape[1] if "input_ids" in batch else 0
        text = self._processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
        return text.strip()

class MOSSASRDiarizeAdapter(_AudioLLMAsrMixin, BaseAdapter):
    """OpenMOSS 转写 + 说话人分离联合模型（trust_remote_code）。

    官方仓库：https://github.com/OpenMOSS/MOSS
    """

    MODELS = ('OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device_obj = torch.device(self.device)
        self._processor = AutoProcessor.from_pretrained(self._load_ref, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._load_ref,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = self._transcribe_audio(inputs, params)
        # 分段解析依赖 MOSS 输出格式（可能为带说话人标记的结构化文本）；
        # TODO: 按官方 README 校准 segments 解析与 text 清理
        segments = self._parse_segments(text)
        if segments is not None:
            # 分段命中时 text 中保留说话人标记原文，由调用方决定是否使用
            text = text
        return {"text": text, "segments": segments, "language": params.get("language")}

    @staticmethod
    def _parse_segments(text: str) -> list[dict[str, Any]] | None:
        """尽力解析说话人分段；无法解析时返回 None。

        示例格式（待按 README 校准）："<|speaker_0|> 你好 <|speaker_1|> 再见"
        """
        if "<|speaker_" not in text:
            return None
        segments: list[dict[str, Any]] = []
        try:
            parts = text.split("<|speaker_")
            for part in parts[1:]:
                spk, _, seg_text = part.partition("|>")
                segments.append({"speaker": f"speaker_{spk.strip()}", "text": seg_text.strip()})
        except (ValueError, IndexError):
            return None
        return segments

    category = 'asr'



# 统一适配器入口（server.py 加载此名称）
Adapter = MOSSASRDiarizeAdapter
