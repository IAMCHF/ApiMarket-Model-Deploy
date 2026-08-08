"""
pyannote/speaker-diarization-community-1 专用适配器（自动生成，请勿手工编辑）。
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

class PyannoteDiarizationAdapter(BaseAdapter):
    """pyannote.audio 说话人分离：多人对话 -> 按说话人切分的时间段。

    注：模型为 gated 仓库，需配置 config.access_token（或环境变量 HF_TOKEN）。
    """

    MODELS = ('pyannote/speaker-diarization-community-1',)

    def _load(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        token = self.config.get("access_token")
        # use_auth_token 传 None 时使用 huggingface_hub 缓存/环境变量中的令牌
        self._pipeline = Pipeline.from_pretrained(self._load_ref, use_auth_token=token or None)
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

    category = 'asr'



# 统一适配器入口（server.py 加载此名称）
Adapter = PyannoteDiarizationAdapter
