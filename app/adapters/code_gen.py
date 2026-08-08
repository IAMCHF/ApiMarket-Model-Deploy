# ============================================================================
# 代码生成（Code Generation）适配器（3 个模型）
#
# 两个适配器族：
#   1. LlamaCppCodeGenAdapter      —— GGUF 量化模型（llama-cpp-python）
#      （Qwen/Qwen2.5-Coder-7B-Instruct-GGUF）
#   2. TransformersCodeGenAdapter  —— MoE 代码大模型（transformers + fp16）
#      （Qwen/Qwen3-Coder-30B-A3B-Instruct / deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct）
#
# /predict 输入输出约定：
#   输入：{"prompt": "..."} 或 {"prompt": "...", "system": "..."} 或 {"messages": [...]}
#   输出：{"text": "..."}  （GGUF 额外返回 {"finish_reason": "..."}）
# ============================================================================

import os
from typing import Any

from .base import BaseAdapter, InvalidInputError, ModelLoadError
from .registry import register_adapter


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


@register_adapter(category="code-gen")
class LlamaCppCodeGenAdapter(BaseAdapter):
    """GGUF 量化代码生成模型（llama-cpp-python 本地推理）。"""

    MODELS = (
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
    )

    def _load(self) -> None:
        """延迟导入 llama_cpp，从模型根目录加载 .gguf 权重。

        权重约定：模型文件存放于 $MODELS_ROOT/<HF_REPO_ID>/ 下（由
        tools/download_models.py 拉取，MODELS_ROOT 默认 /data/models）。
        若目录中找不到 .gguf 文件，给出下载提示而非静默失败。
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：llama-cpp-python 未安装，无法加载 GGUF 模型（{exc}）"
            ) from exc

        models_root = os.environ.get("MODELS_ROOT", "/data/models")
        model_dir = os.path.join(models_root, self.model_id)

        # 显式指定权重路径优先（config["gguf_file"]），否则扫描目录中的 *.gguf
        gguf_path = self.config.get("gguf_file")
        if isinstance(gguf_path, str) and gguf_path:
            gguf_file = gguf_path
        else:
            if not os.path.isdir(model_dir):
                raise ModelLoadError(
                    f"未找到模型目录 {model_dir}：请先运行 tools/download_models.py "
                    f"拉取 {self.model_id} 的 .gguf 权重到 {models_root}。"
                )
            candidates = sorted(
                f for f in os.listdir(model_dir) if f.lower().endswith(".gguf")
            )
            if not candidates:
                raise ModelLoadError(
                    f"模型目录 {model_dir} 中没有 .gguf 权重文件：请使用 "
                    f"tools/download_models.py 下载 {self.model_id} 的量化权重。"
                )
            gguf_file = os.path.join(model_dir, candidates[0])

        n_gpu_layers = int(self.config.get("n_gpu_layers", 0))
        n_ctx = int(self.config.get("n_ctx", 8192))
        self._llm = Llama(
            model_path=gguf_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        messages = _validate_prompt_inputs(inputs)

        max_new_tokens = int(params.get("max_new_tokens", self.config.get("max_new_tokens", 1024)))
        temperature = float(params.get("temperature", self.config.get("temperature", 0.2)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))

        # OpenAI 风格多轮对话 -> 走 create_chat_completion；否则直接续写
        if inputs.get("messages") is not None:
            chat = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choice = chat["choices"][0]
            text = choice["message"]["content"] or ""
            finish_reason = str(choice.get("finish_reason", "stop"))
        else:
            prompt = messages[-1]["content"]
            completion = self._llm.create_completion(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            choice = completion["choices"][0]
            text = choice["text"]
            finish_reason = str(choice.get("finish_reason", "stop"))

        return {"text": text, "finish_reason": finish_reason}


@register_adapter(category="code-gen")
class TransformersCodeGenAdapter(BaseAdapter):
    """transformers 加载的 MoE 代码生成模型（fp16，device 跟随 self.device）。"""

    MODELS = (
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    )

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

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
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
