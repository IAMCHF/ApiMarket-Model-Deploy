"""
openai/privacy-filter 专用适配器（自动生成，请勿手工编辑）。
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


class PrivacyFilterAdapter(BaseAdapter):
    """openai/privacy-filter 隐私 / 敏感信息文本过滤。

    该模型为文本序列分类器，输出 "flagged"（命中隐私/敏感信息）
    与 "safe" 两类标签。
    """

    MODELS = ('openai/privacy-filter',)

    def _load(self) -> None:
        """延迟导入 transformers，加载序列分类模型。"""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：transformers / torch 未安装，无法加载隐私过滤器（{exc}）"
            ) from exc

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._load_ref)
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.nn.functional import softmax

        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("inputs 必须包含非空的 'text'（字符串）")

        encoded = self._tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device_obj)

        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probs = softmax(logits, dim=-1)

        max_idx = int(torch.argmax(logits, dim=-1).item())
        score = float(probs[0, max_idx].item())
        # 优先使用模型 id2label，兜底按 0/1 位置映射
        id2label = getattr(self._model.config, "id2label", None) or {}
        label = str(id2label.get(max_idx, "flagged" if max_idx == 1 else "safe"))
        flagged = "flagged" in label.lower()

        return {"flagged": flagged, "score": score, "label": label}

    category = 'other'



# 统一适配器入口（server.py 加载此名称）
Adapter = PrivacyFilterAdapter
