"""
stepfun-ai/Step1X-Edit 专用适配器（自动生成，请勿手工编辑）。
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


class DiffusersEditAdapter(BaseAdapter):
    """diffusers 图像编辑（stepfun-ai/Step1X-Edit：自然语言指令驱动编辑）。

    注：Step1X-Edit 官方基于图像编辑管线发布，若其权重格式为自定义
    modeling 文件，请按官方 README 替换为对应加载方式（信任远程代码）。
    """

    MODELS = ('stepfun-ai/Step1X-Edit',)

    def _load(self) -> None:
        import torch
        from diffusers import AutoPipelineForImage2Image

        self._device_obj = torch.device(self.device)
        self._pipeline = AutoPipelineForImage2Image.from_pretrained(self._load_ref,
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

    category = 'image-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = DiffusersEditAdapter
