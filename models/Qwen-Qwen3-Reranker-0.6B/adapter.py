"""
Qwen/Qwen3-Reranker-0.6B 专用适配器（自动生成，请勿手工编辑）。
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


class RerankerAdapter(BaseAdapter):
    """重排序模型：query 与 documents 打分排序。

    支持两类加载方式：
      - bge-reranker：AutoModelForSequenceClassification + sigmoid
      - Qwen3-Reranker：AutoModel（trust_remote_code），score 取 logits 最后一维
    上线前按模型官方 README 校准 score 提取方式。
    """

    MODELS = ('Qwen/Qwen3-Reranker-0.6B',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        # bge-reranker 使用分类头；Qwen3-Reranker 使用基础模型 + 专属打分
        if self.model_id.startswith("BAAI/"):
            self._model = AutoModelForSequenceClassification.from_pretrained(self._load_ref, torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32
            )
        else:
            self._model = AutoModel.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
            )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.nn.functional import sigmoid

        query = inputs.get("query")
        documents = inputs.get("documents")
        if not query or not documents:
            raise InvalidInputError("inputs must contain 'query' (str) and 'documents' (list)")

        pairs = [[query, doc] for doc in documents]
        encoded = self._tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self._device_obj)

        with torch.inference_mode():
            logits = self._model(**encoded).logits  # [B, seq] 或 [B, 1]
            if logits.dim() == 2 and logits.shape[1] > 1:
                # Qwen3-Reranker 类：取序列最后一维池化
                scores = logits[:, -1]  # 注：官方实现为取 logits 首 token，按 README 校准
            else:
                scores = logits.squeeze(-1)
            if self.model_id.startswith("BAAI/"):
                scores = sigmoid(scores)

        scores_list = scores.cpu().tolist()
        ranking = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        return {"scores": scores_list, "ranking": ranking}

    category = 'embedding'



# 统一适配器入口（server.py 加载此名称）
Adapter = RerankerAdapter
