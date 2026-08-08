"""
IndexTeam/IndexTTS2 专用适配器（自动生成，请勿手工编辑）。
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


class IndexTTSAdapter(BaseAdapter):
    """IndexTeam IndexTTS2 语音合成（index_tts 包）。

    官方仓库：https://github.com/index-tts/index-tts
    speaker 输入可接受参考音频 base64（音色克隆）或说话人 ID。
    """

    MODELS = ('IndexTeam/IndexTTS2',)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；index_tts 提供 IndexTTS2 类
        from index_tts import IndexTTS2

        self._tts = IndexTTS2(device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        ref_path = None
        if speaker:
            # 参考音频 base64 -> 临时 wav；TODO: 若 speaker 为说话人 ID 则跳过此步
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b64_to_bytes(speaker))
                ref_path = tmp.name
        try:
            # TODO: tts 方法与返回结构（np.ndarray, sr）按官方 README 校准
            audio_np, sr = self._tts.tts(text, ref_audio_path=ref_path)
        finally:
            if ref_path:
                os.unlink(ref_path)
        return {"audio": numpy_audio_to_b64(audio_np, int(sr)), "sample_rate": int(sr), "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = IndexTTSAdapter
