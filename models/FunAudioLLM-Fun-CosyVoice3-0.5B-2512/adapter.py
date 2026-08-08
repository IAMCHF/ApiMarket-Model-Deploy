"""
FunAudioLLM/Fun-CosyVoice3-0.5B-2512 专用适配器（自动生成，请勿手工编辑）。
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


class CosyVoiceAdapter(BaseAdapter):
    """FunASR 加载 CosyVoice3（Fun-AudioLLM）语音合成。

    零样本克隆：传入 inputs.prompt_speech（参考音频 base64）与可选
    params.prompt_text（参考文本）即可克隆音色。
    """

    MODELS = ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',)

    def _load(self) -> None:
        # 延迟导入：funasr 依赖在 requirements/tts.txt 中按需安装
        from funasr import AutoModel

        self._model = AutoModel(
            model=self._load_ref,
            trust_remote_code=True,
            device=self.device if self.device.startswith("cuda") else "cpu",
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        gen_kwargs: dict[str, Any] = {
            "text": text,
            "spk_id": str(inputs.get("speaker") or params.get("speaker") or "中文女"),
            "stream": False,
        }
        prompt_text = inputs.get("prompt_text") or params.get("prompt_text")
        if prompt_text:
            gen_kwargs["prompt_text"] = prompt_text

        prompt_path = None
        prompt_b64 = inputs.get("prompt_speech") or inputs.get("prompt_audio")
        if prompt_b64:
            # CosyVoice 支持 prompt_speech 参考音色；funasr 接受 wav 文件路径
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b64_to_bytes(prompt_b64))
                prompt_path = tmp.name
            gen_kwargs["prompt_speech"] = prompt_path
        try:
            result = self._model.generate(**gen_kwargs)
        finally:
            if prompt_path:
                os.unlink(prompt_path)

        # 兼容生成器返回；离线（stream=False）通常返回 list[dict]
        if hasattr(result, "__next__"):
            result = list(result)
        # TODO: 按官方 README 校准字段名；CosyVoice3 的 result 结构可能与 v2 略有差异
        audio = result[0]["audio"]  # np.ndarray [T]
        sr = int(result[0].get("sample_rate", 24000))
        return {"audio": numpy_audio_to_b64(audio, sr), "sample_rate": sr, "format": "wav"}

    category = 'tts'



# 统一适配器入口（server.py 加载此名称）
Adapter = CosyVoiceAdapter
