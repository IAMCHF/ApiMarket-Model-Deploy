"""
numind/NuExtract3 专用适配器（自动生成，请勿手工编辑）。
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


class NuExtractAdapter(BaseAdapter):
    """numind/NuExtract3：文档信息抽取为结构化 JSON（输入 schema）。"""

    MODELS = ('numind/NuExtract3',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self._load_ref,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import json
        import torch

        document = inputs.get("document")
        schema = inputs.get("schema")
        if not document or not schema:
            raise InvalidInputError("inputs must contain 'document' (str) and 'schema' (dict)")
        if isinstance(schema, dict):
            schema = json.dumps(schema, ensure_ascii=False)

        prompt = (
            f"提取以下文档中符合 schema 的信息，只输出 JSON：\n{schema}\n\n文档：\n{document}"
        )
        encoded = self._tokenizer(prompt, return_tensors="pt").to(self._device_obj)
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=int(params.get("max_new_tokens", 1024)), do_sample=False
            )
        prompt_len = encoded["input_ids"].shape[1]
        text = self._tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
        try:
            extracted = json.loads(text)
        except json.JSONDecodeError:
            extracted = {"raw": text}
        return {"extracted": extracted}

    category = 'ocr'



# 统一适配器入口（server.py 加载此名称）
Adapter = NuExtractAdapter
