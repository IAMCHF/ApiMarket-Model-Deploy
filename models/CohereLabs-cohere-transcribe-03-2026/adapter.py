"""
CohereLabs/cohere-transcribe-03-2026 专用适配器（自动生成，请勿手工编辑）。
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

class NeMoASRAdapter(BaseAdapter):
    """NVIDIA NeMo ASR 生态适配器（流式 / 非流式 / 多语言翻译 / FastConformer）。

    覆盖模型：
      - nvidia/nemotron-3.5-asr-streaming-0.6b  流式 ASR（transcribe 按整段离线处理）
      - nvidia/parakeet-tdt-0.6b-v3             TDT 非流式 ASR
      - nvidia/canary-qwen-2.5b                 ASR + 翻译（transcribe 支持 language 参数）
      - CohereLabs/cohere-transcribe-03-2026    FastConformer（NeMo 格式权重）

    注：canary 系列输出可能带 "eng:" 语言码前缀，_strip_language_prefix 会将其剥离。
    """

    MODELS = ('CohereLabs/cohere-transcribe-03-2026',)

    def _load(self) -> None:
        # 延迟导入：NeMo 依赖在 requirements/asr.txt 中按需安装
        from nemo.collections.asr.models import ASRModel

        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        try:
            self._model = ASRModel.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        except TypeError:
            # 旧版 NeMo 不支持 trust_remote_code 参数，回退默认加载
            self._model = ASRModel.from_pretrained(self._load_ref)
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

    category = 'asr'



# 统一适配器入口（server.py 加载此名称）
Adapter = NeMoASRAdapter
