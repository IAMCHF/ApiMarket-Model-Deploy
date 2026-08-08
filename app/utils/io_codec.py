# ============================================================================
# 输入输出编解码：base64 <-> 图像 / 音频 / 原始字节
#
# 统一约定：/predict 请求中的文件类输入一律使用 base64 字符串；
# 生成类模型的输出（图像/音频/3D/视频）同样以 base64 返回。
# ============================================================================

import base64
import io
from typing import Any


def b64_to_bytes(data: str) -> bytes:
    """base64 字符串 -> 原始字节。"""
    return base64.b64decode(data)


def bytes_to_b64(data: bytes) -> str:
    """原始字节 -> base64 字符串（ASCII）。"""
    return base64.b64encode(data).decode("ascii")


def b64_to_image(data: str) -> Any:
    """base64 -> PIL.Image（RGB）。依赖 Pillow（L2 层已固化）。"""
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def image_to_b64(image: Any, fmt: str = "PNG") -> str:
    """PIL.Image -> base64 字符串。"""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return bytes_to_b64(buf.getvalue())


def b64_to_audio(data: str, sample_rate: int | None = None):
    """base64 -> (waveform: torch.Tensor [C, T], sample_rate)。

    依赖 torchaudio（L2 层已固化）。sample_rate 非空且与源不一致时自动重采样。
    """
    import torchaudio

    buf = io.BytesIO(base64.b64decode(data))
    waveform, sr = torchaudio.load(buf)
    if sample_rate is not None and sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    return waveform, sr


def audio_to_b64(waveform: Any, sample_rate: int, fmt: str = "wav") -> str:
    """torch.Tensor [C, T] -> base64 音频字符串。"""
    import torchaudio

    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format=fmt)
    return bytes_to_b64(buf.getvalue())


def numpy_audio_to_b64(audio_np: Any, sample_rate: int) -> str:
    """numpy 数组 (T,) 或 (C, T) -> base64 音频字符串。"""
    import numpy as np
    import torch
    import torchaudio

    tensor = torch.from_numpy(np.asarray(audio_np)).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, tensor, sample_rate, format="wav")
    return bytes_to_b64(buf.getvalue())
