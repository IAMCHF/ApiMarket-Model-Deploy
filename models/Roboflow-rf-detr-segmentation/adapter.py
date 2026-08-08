"""
Roboflow/rf-detr-segmentation 专用适配器（自动生成，请勿手工编辑）。
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


class RFDetrAdapter(BaseAdapter):
    """Roboflow rf-detr 实时目标检测 / 实例分割。

    优先使用 roboflow `rfdetr` pip 包（RFDETRBase / RFDETRSegmentation）；
    未安装时回退到 transformers 的 DETR 系加载（trust_remote_code，仅目标检测）。
    分割模型（rf-detr-segmentation）额外返回 masks。

    TODO：rfdetr 包的构造参数与 Detection 对象属性（boxes/masks）按官方 README 校准。
    """

    MODELS = ('Roboflow/rf-detr-segmentation',)

    def _load(self) -> None:
        self._device_obj = None
        try:
            from rfdetr import RFDETRBase, RFDETRSegmentation

            model_cls = RFDETRSegmentation if "segmentation" in self.model_id else RFDETRBase
            try:
                self._model = model_cls(device=self.device)
            except TypeError:  # 兼容旧版本构造签名
                self._model = model_cls()
                if self.device.startswith("cuda"):
                    self._model.to(self.device)
            self._backend = "rfdetr"
        except ImportError:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._device_obj = torch.device(self.device)
            self._processor = AutoImageProcessor.from_pretrained(self._load_ref, trust_remote_code=True)
            self._model = AutoModelForObjectDetection.from_pretrained(self._load_ref,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
            )
            self._model.eval().to(self._device_obj)
            self._backend = "transformers"

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        threshold = float(params.get("confidence_threshold", 0.5))

        if self._backend == "rfdetr":
            import numpy as np

            result = self._model.predict(image, threshold=threshold)
            detections: list[dict[str, Any]] = []
            masks: list[str] = []
            for pred in result:
                box = pred.boxes.tolist()
                detections.append(
                    {
                        "box": [float(v) for v in box],
                        "label": pred.class_name,
                        "confidence": round(float(pred.confidence), 4),
                    }
                )
                pred_masks = getattr(pred, "masks", None)
                if pred_masks:
                    for m in pred_masks:
                        masks.append(self._mask_np_to_png(np.asarray(m)))
            return {"detections": detections, "masks": masks or None}

        # transformers 回退：DETR 系目标检测
        import torch

        batch = self._processor(images=image, return_tensors="pt").to(self._device_obj)
        with torch.inference_mode():
            outputs = self._model(**batch)
        results = self._processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=threshold,
        )[0]
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "label": self._model.config.id2label.get(int(label), str(label)),
                    "confidence": round(float(score), 4),
                }
            )
        return {"detections": detections, "masks": None}

    @staticmethod
    def _mask_np_to_png(mask_np: Any) -> str:
        """二值掩码 numpy 数组 -> base64 PNG。"""
        import numpy as np
        from PIL import Image

        arr = mask_np.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        return image_to_b64(img, fmt="PNG")

    category = 'detection'



# 统一适配器入口（server.py 加载此名称）
Adapter = RFDetrAdapter
