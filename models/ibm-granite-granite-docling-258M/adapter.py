"""
ibm-granite/granite-docling-258M 专用适配器（自动生成，请勿手工编辑）。
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


class DoclingAdapter(BaseAdapter):
    """IBM granite-docling：PDF/图片 -> 结构化文档（DocTags/Markdown）。"""

    MODELS = ('ibm-granite/granite-docling-258M',)

    def _load(self) -> None:
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import tempfile

        from server import b64_to_bytes

        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs must contain 'image' (base64 str)")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b64_to_bytes(image_b64))
            tmp_path = tmp.name
        try:
            result = self._converter.convert(tmp_path)
            markdown = result.document.export_to_markdown()
        finally:
            import os

            os.unlink(tmp_path)
        return {"markdown": markdown}

    category = 'ocr'



# 统一适配器入口（server.py 加载此名称）
Adapter = DoclingAdapter
