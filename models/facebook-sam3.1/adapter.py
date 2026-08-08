"""
facebook/sam3.1 专用适配器（自动生成，请勿手工编辑）。
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


class SAM3Adapter(BaseAdapter):
    """facebook SAM3 / SAM3.1 图像分割（transformers 集成，trust_remote_code）。

    输入支持：
      - 仅图像：自动分割（无提示）
      - 图像 + 提示点/提示框：通过 params 传入，坐标使用像素坐标：
          points: [[x, y], [x, y], ...]
          boxes:  [[x1, y1, x2, y2], ...]
          labels: 对应点/框的标签（1 前景 / 0 背景），缺省全为 1

    输出：masks（base64 灰度 PNG 列表）+ 对应的预测分数。

    TODO：SAM3 / SAM3.1 官方集成路径（Sam3 / Sam3Processor 类名、post_process
    参数、多掩码选择策略）按各自 README 校准后上线。
    """

    MODELS = ('facebook/sam3.1',)

    def _load(self) -> None:
        import torch

        try:  # 新版 transformers：Sam3 / Sam3Processor
            from transformers import Sam3, Sam3Processor
        except ImportError:  # 兼容旧版类名（Sam3Model）
            from transformers import Sam3Model as Sam3
            from transformers import Sam3Processor

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._processor = Sam3Processor.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        self._model = Sam3.from_pretrained(self._load_ref,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        # 可选提示点 / 提示框 / 标签（像素坐标）
        points = params.get("points")
        boxes = params.get("boxes")
        labels = params.get("labels")

        proc_input: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if points:
            proc_input["input_points"] = [points]  # [B, N, 2]
        if boxes:
            proc_input["input_boxes"] = [boxes]  # [B, N, 4]
        if labels:
            proc_input["input_labels"] = [labels]
        processed = self._processor(**proc_input).to(self._device_obj)

        with torch.inference_mode():
            outputs = self._model(**processed)

        # post_process_masks：从处理器分辨率还原到原始图像尺寸
        original_size = list(image.size[::-1])  # (H, W)
        input_size = list(processed.pixel_values.shape[-2:])  # 处理器预处理后的 (H', W')
        masks = self._processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[original_size],
            input_sizes=[input_size],
        )[0]  # [num_masks, H, W]
        scores = outputs.iou_scores[0]  # [num_masks]

        # SAM 默认多掩码输出：逐个编码返回（主掩码取置信度最高者置首）
        order = list(range(masks.shape[0]))
        order.sort(key=lambda i: float(scores[i].item()), reverse=True)
        mask_b64 = [self._mask_to_png(masks[i]) for i in order]
        score_list = [round(float(scores[i].item()), 4) for i in order]
        return {"detections": [], "masks": mask_b64, "scores": score_list}

    @staticmethod
    def _mask_to_png(mask: Any) -> str:
        """概率/二值掩码张量 [H, W] -> base64 PNG 字符串。"""
        import numpy as np
        from PIL import Image

        arr = mask.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        return image_to_b64(img, fmt="PNG")

    category = 'detection'



# 统一适配器入口（server.py 加载此名称）
Adapter = SAM3Adapter
