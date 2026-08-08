"""
PaddlePaddle/PaddleOCR-VL-1.6 专用适配器（自动生成，请勿手工编辑）。
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


class PaddleOCRAdapter(BaseAdapter):
    """PaddlePaddle 系 OCR：PaddleOCR-VL（VLM）与 PP-OCRv5（经典检测+识别）。"""

    MODELS = ('PaddlePaddle/PaddleOCR-VL-1.6',)

    def _load(self) -> None:
        # paddlepaddle-gpu 自带 CUDA 运行时，与镜像 CUDA 共存（见方案 8.1）
        from paddleocr import PaddleOCR

        # PaddleOCR-VL 使用 use_doc_orientation_classify 等参数；PP-OCRv5 使用经典参数
        kwargs = dict(
            use_gpu=self.device.startswith("cuda"),
            lang="ch" if "PP-OCRv5" in self.model_id else "chinese",
        )
        if "PP-OCRv5" in self.model_id:
            kwargs.update(dict(det_model_dir=None, rec_model_dir=None))
        else:
            kwargs.update(dict(ocr_version="PP-OCRv5"))
        self._ocr = PaddleOCR(**kwargs)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs must contain 'image' (base64 str)")

        import tempfile

        from server import b64_to_bytes

        # PaddleOCR 接受路径输入；base64 落临时文件后推理
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b64_to_bytes(image_b64))
            tmp_path = tmp.name
        try:
            result = self._ocr.ocr(tmp_path, cls=True)
        finally:
            import os

            os.unlink(tmp_path)

        # result: [[ [box, (text, conf)], ... ]]（分页）
        pages = []
        for page in result or []:
            texts = [(line[1][0], round(float(line[1][1]), 4)) for line in page or [] if line]
            pages.append({"texts": texts, "text": "\n".join(t[0] for t in texts)})
        return {"pages": pages, "text": "\n".join(p["text"] for p in pages)}

    category = 'ocr'



# 统一适配器入口（server.py 加载此名称）
Adapter = PaddleOCRAdapter
