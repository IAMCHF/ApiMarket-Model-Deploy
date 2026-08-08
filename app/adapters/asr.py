# ============================================================================
# 语音识别（ASR）与说话人分离适配器（9 个模型）
#
# 四个适配器族：
#   1. NeMoASRAdapter             —— NVIDIA NeMo ASR 生态
#        nvidia/nemotron-3.5-asr-streaming-0.6b（流式）/ parakeet-tdt-0.6b-v3
#        / canary-qwen-2.5b（ASR+翻译）/ CohereLabs FastConformer
#   2. PyannoteDiarizationAdapter —— pyannote.audio 说话人分离
#   3. HuggingFaceASRAdapter      —— transformers 音频大模型
#        Qwen3-ASR-1.7B / Qwen3-ASR-0.6B-hf / Step-Audio-2-mini（trust_remote_code）
#   4. MOSSASRDiarizeAdapter      —— MOSS 转写+说话人分离联合模型（trust_remote_code）
#
# /predict 输入输出约定：
#   {"audio": "<base64 wav/mp3>"}
#     -> {"text": str, "segments": [...]|None, "language": str|None}
#   - segments 为时间戳分段（ASR 请求 params.timestamps=true 时返回），
#     说话人分离恒返回 segments，且每段含 start / end / speaker 字段。
#   - language 取请求 params.language；canary 类模型可自动识别语言码。
# ============================================================================

import os
import tempfile
from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter
from ..utils.io_codec import b64_to_audio


def _audio_b64_to_temp_wav(audio_b64: str, sample_rate: int = 16000) -> str:
    """base64 音频 -> 临时 16k 单声道 wav 文件路径。

    供 NeMo / pyannote 等仅接受文件路径的库使用；
    调用方须在 finally 中删除返回的临时文件。
    """
    import torchaudio

    waveform, sr = b64_to_audio(audio_b64, sample_rate=sample_rate)
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # 混为单声道
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, waveform, sr, format="wav")
    return path


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


@register_adapter(category="asr")
class NeMoASRAdapter(BaseAdapter):
    """NVIDIA NeMo ASR 生态适配器（流式 / 非流式 / 多语言翻译 / FastConformer）。

    覆盖模型：
      - nvidia/nemotron-3.5-asr-streaming-0.6b  流式 ASR（transcribe 按整段离线处理）
      - nvidia/parakeet-tdt-0.6b-v3             TDT 非流式 ASR
      - nvidia/canary-qwen-2.5b                 ASR + 翻译（transcribe 支持 language 参数）
      - CohereLabs/cohere-transcribe-03-2026    FastConformer（NeMo 格式权重）

    注：canary 系列输出可能带 "eng:" 语言码前缀，_strip_language_prefix 会将其剥离。
    """

    MODELS = (
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/canary-qwen-2.5b",
        "CohereLabs/cohere-transcribe-03-2026",
    )

    def _load(self) -> None:
        # 延迟导入：NeMo 依赖在 requirements/asr.txt 中按需安装
        from nemo.collections.asr.models import ASRModel

        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        try:
            self._model = ASRModel.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        except TypeError:
            # 旧版 NeMo 不支持 trust_remote_code 参数，回退默认加载
            self._model = ASRModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        audio_b64 = inputs.get("audio")
        if not audio_b64:
            raise InvalidInputError("inputs 必须包含 'audio'（base64 音频字符串）")

        language = params.get("language")
        transcribe_kwargs: dict[str, Any] = {
            "batch_size": int(params.get("batch_size", 1)),
            "verbose": False,
        }
        if language:
            transcribe_kwargs["language"] = language
        if params.get("timestamps"):
            transcribe_kwargs["return_hypotheses"] = True

        path = _audio_b64_to_temp_wav(audio_b64)
        try:
            try:
                results = self._model.transcribe([path], **transcribe_kwargs)
            except TypeError:
                # 该模型不支持 language / return_hypotheses 等参数时回退
                results = self._model.transcribe([path], batch_size=1, verbose=False)
        finally:
            os.unlink(path)

        # 兼容返回 str 或 NeMo Hypothesis（return_hypotheses=True）
        first = results[0] if isinstance(results, (list, tuple)) else results
        raw_text = first.text if hasattr(first, "text") else str(first)
        text = self._strip_language_prefix(raw_text)

        segments = None
        if params.get("timestamps"):
            segments = self._hypothesis_to_segments(first)
        return {"text": text, "segments": segments, "language": language or self._detect_language(raw_text)}

    @staticmethod
    def _strip_language_prefix(text: str) -> str:
        """剥离 canary 类输出首部的 "eng:" 语言码前缀。"""
        head, sep, rest = text.partition(":")
        if sep and head.strip().isalpha() and len(head.strip()) <= 4:
            return rest.strip()
        return text

    def _detect_language(self, raw_text: str) -> str | None:
        """从 canary 输出首部提取语言码；其它模型返回 None。"""
        if "canary" not in self.model_id.lower():
            return None
        head, sep, _ = raw_text.partition(":")
        if sep and head.strip().isalpha() and len(head.strip()) <= 4:
            return head.strip()
        return None

    @staticmethod
    def _hypothesis_to_segments(hyp: Any) -> list[dict[str, Any]] | None:
        """NeMo Hypothesis.timestep -> [{"start", "end"}]；结构不符返回 None。"""
        timestep = getattr(hyp, "timestep", None)
        if timestep is None:
            return None
        try:
            starts = getattr(timestep, "start", None)
            ends = getattr(timestep, "end", None)
            if starts is None or ends is None:
                return None
            return [
                {"start": round(float(s), 3), "end": round(float(e), 3)}
                for s, e in zip(starts, ends)
            ]
        except (TypeError, ValueError):
            return None


@register_adapter(category="asr")
class PyannoteDiarizationAdapter(BaseAdapter):
    """pyannote.audio 说话人分离：多人对话 -> 按说话人切分的时间段。

    注：模型为 gated 仓库，需配置 config.access_token（或环境变量 HF_TOKEN）。
    """

    MODELS = ("pyannote/speaker-diarization-community-1",)

    def _load(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        token = self.config.get("access_token")
        # use_auth_token 传 None 时使用 huggingface_hub 缓存/环境变量中的令牌
        self._pipeline = Pipeline.from_pretrained(self.model_id, use_auth_token=token or None)
        self._pipeline.to(torch.device(self.device))

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        audio_b64 = inputs.get("audio")
        if not audio_b64:
            raise InvalidInputError("inputs 必须包含 'audio'（base64 音频字符串）")

        path = _audio_b64_to_temp_wav(audio_b64)
        try:
            diarization = self._pipeline(path)
        finally:
            os.unlink(path)

        segments = [
            {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        # 说话人分离不产出转写文本，text 置空串以满足 I/O 契约
        return {"text": "", "segments": segments, "language": None}


@register_adapter(category="asr")
class HuggingFaceASRAdapter(_AudioLLMAsrMixin, BaseAdapter):
    """transformers 音频大模型 ASR（Qwen3-ASR / Step-Audio-2）。

    注：Step-Audio-2-mini 依赖 trust_remote_code（自定义 modeling 文件）；
    各模型 generate 参数细节按官方 README 校准。
    """

    MODELS = (
        "Qwen/Qwen3-ASR-1.7B",
        "Qwen/Qwen3-ASR-0.6B-hf",
        "stepfun-ai/Step-Audio-2-mini",
    )

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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


@register_adapter(category="asr")
class MOSSASRDiarizeAdapter(_AudioLLMAsrMixin, BaseAdapter):
    """OpenMOSS 转写 + 说话人分离联合模型（trust_remote_code）。

    官方仓库：https://github.com/OpenMOSS/MOSS
    """

    MODELS = ("OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B",)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device_obj = torch.device(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
