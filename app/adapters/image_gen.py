# ============================================================================
# 图像生成 / 编辑适配器（9 个模型）
#
# 全部基于 diffusers（AutoPipeline 自动选择 FLUX/SD3.5/Sana/Ideogram 对应管线）：
#   1. DiffusersT2IAdapter  —— 文生图（8 个模型）
#   2. DiffusersEditAdapter —— 图像编辑（stepfun-ai/Step1X-Edit）
#
# /predict 输入输出约定：
#   文生图：{"prompt": "...", "negative_prompt": "..."} -> {"image": "<base64>"}
#   图像编辑：{"prompt": "...", "image": "<base64>"}     -> {"image": "<base64>"}
# ============================================================================

from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter
from ..utils.io_codec import b64_to_image, image_to_b64


@register_adapter(category="image-gen")
class DiffusersT2IAdapter(BaseAdapter):
    """diffusers 文生图（FLUX.2-klein / SD3.5 / Sana / Ideogram-4）。"""

    MODELS = (
        "ideogram-ai/ideogram-4-fp8",
        "ideogram-ai/ideogram-4-nf4",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-4B",
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "stabilityai/stable-diffusion-3.5-medium",
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px",
        "Efficient-Large-Model/Sana_1.5",
    )

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
        self._pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id, **kwargs)
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


@register_adapter(category="image-gen")
class DiffusersEditAdapter(BaseAdapter):
    """diffusers 图像编辑（stepfun-ai/Step1X-Edit：自然语言指令驱动编辑）。

    注：Step1X-Edit 官方基于图像编辑管线发布，若其权重格式为自定义
    modeling 文件，请按官方 README 替换为对应加载方式（信任远程代码）。
    """

    MODELS = ("stepfun-ai/Step1X-Edit",)

    def _load(self) -> None:
        import torch
        from diffusers import AutoPipelineForImage2Image

        self._device_obj = torch.device(self.device)
        self._pipeline = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self._pipeline.to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        prompt = inputs.get("prompt")
        image_b64 = inputs.get("image")
        if not prompt or not image_b64:
            raise InvalidInputError("inputs must contain 'prompt' (str) and 'image' (base64 str)")

        image = b64_to_image(image_b64)
        result = self._pipeline(
            prompt=prompt,
            image=image,
            strength=float(params.get("strength", 0.6)),
            num_inference_steps=int(params.get("num_inference_steps", 30)),
            guidance_scale=float(params.get("guidance_scale", 3.5)),
        ).images[0]
        return {"image": image_to_b64(result, fmt="PNG")}
