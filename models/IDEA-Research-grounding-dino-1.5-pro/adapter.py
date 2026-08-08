"""
IDEA-Research/grounding-dino-1.5-pro 专用适配器（自动生成，请勿手工编辑）。
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


class GroundingDINOAdapter(BaseAdapter):
    """IDEA-Research grounding-dino 开放词表目标检测。

    使用 groundingdino 官方仓库推理（groundingdino.util.inference），
    文本提示词通过 inputs.prompt 或 params.prompt 传入。
    权重与配置从 HF 仓库快照中定位后交给官方 load_model。

    TODO：config/权重文件定位规则与 predict 的图像格式（RGB/BGR）
    按 grounding-dino-1.5-pro 官方 README 校准后上线。
    """

    MODELS = ('IDEA-Research/grounding-dino-1.5-pro',)

    def _load(self) -> None:
        import glob
        import os

        import torch
        from groundingdino.util.inference import load_model
        from huggingface_hub import snapshot_download

        self._device_obj = torch.device(self.device)
        repo_dir = self._weights_dir if self._weights_dir else snapshot_download(repo_id=self.model_id)
        config_path = self._locate_config(repo_dir)
        weights_path = self._locate_weights(repo_dir)
        if not config_path or not weights_path:
            raise RuntimeError(f"无法在仓库 {self.model_id} 中定位 GroundingDINO 的 config 或权重文件")
        self._model = load_model(config_path, weights_path, device=self.device)

    @staticmethod
    def _locate_config(repo_dir: str) -> str | None:
        import glob
        import os

        for pattern in ("*.yaml", "*.yml", "config.py", "config.json"):
            hits = glob.glob(os.path.join(repo_dir, "**", pattern), recursive=True)
            if hits:
                return hits[0]
        return None

    @staticmethod
    def _locate_weights(repo_dir: str) -> str | None:
        import glob
        import os

        for pattern in ("*.pth", "*.safetensors", "*.pt"):
            hits = glob.glob(os.path.join(repo_dir, "**", pattern), recursive=True)
            if hits:
                return hits[0]
        return None

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from groundingdino.util.inference import predict

        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        caption = inputs.get("prompt") or params.get("prompt")
        if not caption:
            raise InvalidInputError("开放词表检测必须提供文本提示：inputs.prompt 或 params.prompt")

        image = b64_to_image(image_b64)
        boxes, logits, phrases = predict(
            model=self._model,
            image=np.asarray(image),  # 官方 predict 接受 numpy (H, W, 3)
            caption=caption,
            box_threshold=float(params.get("box_threshold", 0.35)),
            text_threshold=float(params.get("text_threshold", 0.25)),
        )

        w, h = image.size
        detections: list[dict[str, Any]] = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            x1, y1, x2, y2 = box.tolist()  # 归一化坐标 [0, 1] -> 像素坐标
            detections.append(
                {
                    "box": [round(x1 * w, 1), round(y1 * h, 1), round(x2 * w, 1), round(y2 * h, 1)],
                    "label": phrase,
                    "confidence": round(float(logit), 4),
                }
            )
        return {"detections": detections, "masks": None}

    category = 'detection'



# 统一适配器入口（server.py 加载此名称）
Adapter = GroundingDINOAdapter
