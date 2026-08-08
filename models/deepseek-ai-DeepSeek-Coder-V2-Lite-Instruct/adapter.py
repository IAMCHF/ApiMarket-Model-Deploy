"""
deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct 专用适配器（自动生成，请勿手工编辑）。
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


def _validate_prompt_inputs(inputs: dict[str, Any]) -> list[dict[str, str]]:
    """将 inputs 归一化为 OpenAI 风格 messages 列表，供聊天模板 / 聊天接口使用。

    支持两种形态：
      - {"messages": [{"role": ..., "content": ...}, ...]}
      - {"prompt": "...", "system": "..." | None}   （system 可选）
    校验失败抛 InvalidInputError（用户可见中文消息）。
    """
    if "messages" in inputs:
        messages = inputs["messages"]
        if not isinstance(messages, list) or not messages:
            raise InvalidInputError("inputs['messages'] 必须是非空列表")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise InvalidInputError(f"inputs['messages'][{i}] 必须是字典")
            if msg.get("role") not in ("system", "user", "assistant"):
                raise InvalidInputError(f"inputs['messages'][{i}] 的 role 非法（仅支持 system/user/assistant）")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise InvalidInputError(f"inputs['messages'][{i}] 的 content 必须是非空字符串")
        return messages

    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InvalidInputError("inputs 必须包含非空的 'prompt'（字符串），或使用 'messages' 列表")

    system = inputs.get("system")
    messages: list[dict[str, str]] = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages

class TransformersCodeGenAdapter(BaseAdapter):
    """transformers 加载的 MoE 代码生成模型（fp16，device 跟随 self.device）。"""

    MODELS = ('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',)

    def _load(self) -> None:
        """延迟导入 transformers，加载 AutoModelForCausalLM + AutoTokenizer。"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：transformers / torch 未安装，无法加载代码模型（{exc}）"
            ) from exc

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", False))
        torch_dtype = (
            torch.float16
            if self._device_obj.type == "cuda"
            else torch.float32
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(self._load_ref,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        messages = _validate_prompt_inputs(inputs)

        max_new_tokens = int(params.get("max_new_tokens", self.config.get("max_new_tokens", 1024)))
        temperature = float(params.get("temperature", self.config.get("temperature", 0.2)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))

        # 通过 chat_template 将 messages 渲染为模型输入；无模板时报错提示
        if self._tokenizer.chat_template is None:
            raise ModelLoadError(
                f"模型 {self.model_id} 未提供 chat_template，请传入可直接续写的 prompt。"
            )
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(prompt_text, return_tensors="pt").to(self._device_obj)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0, encoded["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return {"text": text}

    category = 'code-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = TransformersCodeGenAdapter
