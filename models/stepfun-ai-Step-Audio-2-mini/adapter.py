"""
stepfun-ai/Step-Audio-2-mini 专用适配器（自动生成，请勿手工编辑）。
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

class HuggingFaceASRAdapter(_AudioLLMAsrMixin, BaseAdapter):
    """transformers 音频大模型 ASR（Qwen3-ASR / Step-Audio-2）。

    注：Step-Audio-2-mini 依赖 trust_remote_code（自定义 modeling 文件）；
    各模型 generate 参数细节按官方 README 校准。
    """

    MODELS = ('stepfun-ai/Step-Audio-2-mini',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._processor = AutoProcessor.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(self._load_ref,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = self._transcribe_audio(inputs, params)
        # Qwen3-ASR 输出可能带 <|en|> 语言 token 前缀，提取并剥离
        language = params.get("language") or self._detect_language_token(text)
        if language and text.startswith(f"<|{language}|>"):
            text = text[len(f"<|{language}|>"):].strip()
        return {"text": text, "segments": None, "language": language}

    @staticmethod
    def _detect_language_token(text: str) -> str | None:
        """从 "<|xx|>" 形式的前缀提取语言码；无则返回 None。"""
        if text.startswith("<|") and "|>" in text:
            token = text[2:].partition("|>")[0].strip()
            if token.isalpha() and len(token) <= 4:
                return token
        return None

    category = 'asr'



# 统一适配器入口（server.py 加载此名称）
Adapter = HuggingFaceASRAdapter
