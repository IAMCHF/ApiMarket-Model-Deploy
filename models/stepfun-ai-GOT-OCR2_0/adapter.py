"""
stepfun-ai/GOT-OCR2_0 专用适配器（自动生成，请勿手工编辑）。
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


class GOTOCRAdapter(BaseAdapter):
    """stepfun-ai/GOT-OCR2_0：端到端通用格式化细粒度 OCR。"""

    MODELS = ('stepfun-ai/GOT-OCR2_0',)

    def _load(self) -> None:
        # 官方推理：transformers + trust_remote_code（GOT 专用 modeling 文件）
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self._load_ref,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import tempfile

        from server import b64_to_bytes

        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs must contain 'image' (base64 str)")

        # GOT-OCR2 支持 ocr / format / fine-grained 等模式
        mode = params.get("mode", "format")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b64_to_bytes(image_b64))
            tmp_path = tmp.name
        try:
            text = self._model.chat(self._tokenizer, tmp_path, ocr_type="format" if mode == "markdown" else mode)
        finally:
            import os

            os.unlink(tmp_path)
        return {"text": text}

    category = 'ocr'



# 统一适配器入口（server.py 加载此名称）
Adapter = GOTOCRAdapter
