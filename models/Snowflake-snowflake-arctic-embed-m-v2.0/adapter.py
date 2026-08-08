"""
Snowflake/snowflake-arctic-embed-m-v2.0 专用适配器（自动生成，请勿手工编辑）。
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


class SentenceTransformerAdapter(BaseAdapter):
    """sentence-transformers 生态嵌入模型。"""

    MODELS = ('Snowflake/snowflake-arctic-embed-m-v2.0',)

    def _load(self) -> None:
        # 延迟导入：依赖在 requirements/embedding.txt 中按需安装
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._load_ref, device=self.device)

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

    category = 'embedding'



# 统一适配器入口（server.py 加载此名称）
Adapter = SentenceTransformerAdapter
