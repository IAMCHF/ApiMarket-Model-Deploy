"""
briaai/RMBG-2.0 专用适配器（自动生成，请勿手工编辑）。
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


class RMBGAdapter(BaseAdapter):
    """briaai/RMBG-2.0 背景移除（transformers AutoModelForImageSegmentation）。

    使用 briaai 官方预处理（Resize(1024) + ImageNet 归一化），
    取输出 output[0][0] 作为前景概率掩码。返回：
      - image：移除背景后的 RGBA 透明 PNG
      - mask ：前景掩码（灰度 PNG）
    """

    MODELS = ('briaai/RMBG-2.0',)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        self._device_obj = torch.device(self.device)
        self._model = AutoModelForImageSegmentation.from_pretrained(self._load_ref,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms

        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        # briaai 官方预处理（与 RMBG-2.0 官方 demo 一致）
        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(image).unsqueeze(0).to(self._device_obj)
        with torch.inference_mode():
            preds = self._model(tensor)[0][0]  # output[0][0]：前景概率掩码 [1024, 1024]

        mask_1024 = preds.squeeze().cpu().float().numpy()
        mask_img = Image.fromarray((np.clip(mask_1024, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
        mask_img = mask_img.resize(image.size, Image.BILINEAR)  # 还原到原图尺寸

        # 将掩码作为 alpha 通道合成透明背景 PNG
        rgba = image.convert("RGBA")
        rgba.putalpha(mask_img)
        mask_b64 = image_to_b64(mask_img, fmt="PNG")
        return {
            "image": image_to_b64(rgba, fmt="PNG"),
            "mask": mask_b64,
            "detections": [],
            "masks": [mask_b64],
        }

    category = 'detection'



# 统一适配器入口（server.py 加载此名称）
Adapter = RMBGAdapter
