"""
Efficient-Large-Model/Sana_Sprint_1.6B_1024px 专用适配器（自动生成，请勿手工编辑）。
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


class DiffusersT2IAdapter(BaseAdapter):
    """diffusers 文生图（FLUX.2-klein / SD3.5 / Sana / Ideogram-4）。"""

    MODELS = ('Efficient-Large-Model/Sana_Sprint_1.6B_1024px',)

    # 模型级默认生成参数（可被请求 params 覆盖）
    DEFAULTS: dict[str, dict[str, Any]] = {
        "ideogram-ai/ideogram-4-fp8": {"width": 1024, "height": 1024, "num_inference_steps": 40},
        "ideogram-ai/ideogram-4-nf4": {"width": 1024, "height": 1024, "num_inference_steps": 40},
        "black-forest-labs/FLUX.2-klein-9B": {"width": 1024, "height": 1024, "num_inference_steps": 30},
        "black-forest-labs/FLUX.2-klein-4B": {"width": 1024, "height": 1024, "num_inference_steps": 30},
        "stabilityai/stable-diffusion-3.5-large-turbo": {"width": 1024, "height": 1024, "num_inference_steps": 4},
        "stabilityai/stable-diffusion-3.5-medium": {"width": 1024, "height": 1024, "num_inference_steps": 28},
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px": {"width": 1024, "height": 1024, "num_inference_steps": 1},
        "Efficient-Large-Model/Sana_1.5": {"width": 1024, "height": 1024, "num_inference_steps": 20},
    }

    def _load(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image

        self._device_obj = torch.device(self.device)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        }
        # 量化变体：fp8/nf4 仓库直接加载（对应 variant 或量化权重）
        if "ideogram-4-fp8" in self.model_id:
            kwargs["variant"] = "fp8"
        if "ideogram-4-nf4" in self.model_id:
            kwargs["quantization_config"] = None  # 直接加载 NF4 权重（若需 bitsandbytes 则启用）
        self._pipeline = AutoPipelineForText2Image.from_pretrained(self._load_ref, **kwargs)
        self._pipeline.to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        prompt = inputs.get("prompt")
        if not prompt:
            raise InvalidInputError("inputs must contain 'prompt' (str)")

        defaults = self.DEFAULTS.get(self.model_id, {})
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": inputs.get("negative_prompt", params.get("negative_prompt", "")),
            "width": int(params.get("width", defaults.get("width", 1024))),
            "height": int(params.get("height", defaults.get("height", 1024))),
            "num_inference_steps": int(params.get("num_inference_steps", defaults.get("num_inference_steps", 30))),
        }
        if params.get("guidance_scale") is not None:
            gen_kwargs["guidance_scale"] = float(params["guidance_scale"])
        if params.get("seed") is not None:
            import torch

            torch.manual_seed(int(params["seed"]))

        image = self._pipeline(**gen_kwargs).images[0]
        return {"image": image_to_b64(image, fmt="PNG")}

    category = 'image-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = DiffusersT2IAdapter
