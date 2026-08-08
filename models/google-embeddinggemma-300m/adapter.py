"""
google/embeddinggemma-300m 专用适配器（自动生成，请勿手工编辑）。
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


class _PoolingMixin:
    """transformers 嵌入模型通用的 last_hidden_state mean pooling + 归一化。"""

    @staticmethod
    def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
        import torch

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _normalize(vector: Any) -> Any:
        import torch

        return torch.nn.functional.normalize(vector, p=2, dim=-1)

class HuggingFaceEmbeddingAdapter(_PoolingMixin, BaseAdapter):
    """transformers 加载的嵌入模型（AutoModel + mean pooling + L2 norm）。

    注：部分模型（如 Qwen3-Embedding / Gemma-Embedding）要求 trust_remote_code=True；
    个别厂商对 pooling 有专属约定（如 Qwen3 官方使用 last token 附近池化），
    上线前请按模型官方 README 校准 pooling 方式。
    """

    MODELS = ('google/embeddinggemma-300m',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        self._model = AutoModel.from_pretrained(self._load_ref,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        if "texts" in inputs:
            texts = inputs["texts"]
        elif "text" in inputs:
            texts = [inputs["text"]]
        else:
            raise InvalidInputError("inputs must contain 'texts' (list) or 'text' (str)")

        max_length = int(params.get("max_length", self.config.get("max_length", 8192)))
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(self._device_obj)

        with torch.inference_mode():
            outputs = self._model(**encoded)
            # jina v5 多模态模型（文本/图像）返回 last_hidden_state，取文本序列池化
            hidden = outputs.last_hidden_state
            pooled = self._mean_pool(hidden, encoded["attention_mask"])
            pooled = self._normalize(pooled)

        return {
            "embeddings": pooled.cpu().tolist(),
            "dim": int(pooled.shape[1]),
            "count": len(texts),
        }

    category = 'embedding'



# 统一适配器入口（server.py 加载此名称）
Adapter = HuggingFaceEmbeddingAdapter
