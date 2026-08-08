# ============================================================================
# 嵌入检索 / 重排序适配器（17 个模型）
#
# 三个适配器族：
#   1. SentenceTransformerAdapter  —— sentence-transformers 加载（MiniLM/bge-m3/nomic/arctic）
#   2. HuggingFaceEmbeddingAdapter —— transformers 加载 + mean pooling + L2 归一化
#                                   （Nemotron-Embed/Qwen3-Embedding/jina/gemma/granite）
#   3. RerankerAdapter             —— 重排序（bge-reranker/Qwen3-Reranker）
#
# /predict 输入输出约定：
#   嵌入：{"texts": ["..."]} 或 {"text": "..."}
#        -> {"embeddings": [[0.1, ...], ...], "dim": 1024, "count": n}
#   重排序：{"query": "...", "documents": ["..."]}
#        -> {"scores": [0.95, 0.31, ...], "ranking": [0, 2, 1]}
# ============================================================================

from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter


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


@register_adapter(category="embedding")
class SentenceTransformerAdapter(BaseAdapter):
    """sentence-transformers 生态嵌入模型。"""

    MODELS = (
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-m3",
        "nomic-ai/nomic-embed-text-v2-moe",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
    )

    def _load(self) -> None:
        # 延迟导入：依赖在 requirements/embedding.txt 中按需安装
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        if "texts" in inputs:
            texts = inputs["texts"]
        elif "text" in inputs:
            texts = [inputs["text"]]
        else:
            raise InvalidInputError("inputs must contain 'texts' (list) or 'text' (str)")

        batch_size = int(params.get("batch_size", 32))
        embeddings = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return {
            "embeddings": embeddings.tolist(),
            "dim": int(embeddings.shape[1]),
            "count": len(texts),
        }


@register_adapter(category="embedding")
class HuggingFaceEmbeddingAdapter(_PoolingMixin, BaseAdapter):
    """transformers 加载的嵌入模型（AutoModel + mean pooling + L2 norm）。

    注：部分模型（如 Qwen3-Embedding / Gemma-Embedding）要求 trust_remote_code=True；
    个别厂商对 pooling 有专属约定（如 Qwen3 官方使用 last token 附近池化），
    上线前请按模型官方 README 校准 pooling 方式。
    """

    MODELS = (
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "nvidia/Nemotron-3-Embed-8B-BF16",
        "nvidia/Nemotron-3-Embed-1B-NVFP4",
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "jinaai/jina-embeddings-v5-omni-small",
        "google/embeddinggemma-300m",
        "ibm-granite/granite-embedding-97m-multilingual-r2",
    )

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        self._model = AutoModel.from_pretrained(
            self.model_id,
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


@register_adapter(category="embedding")
class RerankerAdapter(BaseAdapter):
    """重排序模型：query 与 documents 打分排序。

    支持两类加载方式：
      - bge-reranker：AutoModelForSequenceClassification + sigmoid
      - Qwen3-Reranker：AutoModel（trust_remote_code），score 取 logits 最后一维
    上线前按模型官方 README 校准 score 提取方式。
    """

    MODELS = (
        "BAAI/bge-reranker-v2-m3",
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B",
    )

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        # bge-reranker 使用分类头；Qwen3-Reranker 使用基础模型 + 专属打分
        if self.model_id.startswith("BAAI/"):
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32
            )
        else:
            self._model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=trust_remote_code,
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
