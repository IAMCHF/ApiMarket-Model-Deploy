# ============================================================================
# 检测 / 分割 / 抠图适配器（8 个模型）
#
# 六个适配器族：
#   1. SAM3Adapter        —— facebook/sam3 / sam3.1 图像分割（transformers 集成）
#   2. RMBGAdapter        —— briaai/RMBG-2.0 背景移除
#   3. RFDetrAdapter      —— Roboflow rf-detr 实时目标检测 / 实例分割
#   4. GroundingDINOAdapter —— IDEA-Research grounding-dino 开放词表检测
#   5. EdgeTAMAdapter     —— facebook/EdgeTAM 实时视频分割（骨架，待接入）
#   6. DEIMAdapter        —— Intellindust/DEIMv2 实时检测（骨架，待接入）
#
# /predict 输入输出约定：
#   输入：{"image": "<base64>", "prompt": "<开放词表文本（可选）>"}
#   输出：{"detections": [{"box": [x1,y1,x2,y2], "label": "...", "confidence": 0.9}, ...],
#         "masks": ["<base64 png>", ...] | None}
# ============================================================================

from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter
from ..utils.io_codec import b64_to_image, image_to_b64


@register_adapter(category="detection")
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

    MODELS = (
        "facebook/sam3",
        "facebook/sam3.1",
    )

    def _load(self) -> None:
        import torch

        try:  # 新版 transformers：Sam3 / Sam3Processor
            from transformers import Sam3, Sam3Processor
        except ImportError:  # 兼容旧版类名（Sam3Model）
            from transformers import Sam3Model as Sam3
            from transformers import Sam3Processor

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        self._processor = Sam3Processor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        self._model = Sam3.from_pretrained(
            self.model_id,
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


@register_adapter(category="detection")
class RMBGAdapter(BaseAdapter):
    """briaai/RMBG-2.0 背景移除（transformers AutoModelForImageSegmentation）。

    使用 briaai 官方预处理（Resize(1024) + ImageNet 归一化），
    取输出 output[0][0] 作为前景概率掩码。返回：
      - image：移除背景后的 RGBA 透明 PNG
      - mask ：前景掩码（灰度 PNG）
    """

    MODELS = ("briaai/RMBG-2.0",)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageSegmentation

        self._device_obj = torch.device(self.device)
        self._model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id,
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


@register_adapter(category="detection")
class RFDetrAdapter(BaseAdapter):
    """Roboflow rf-detr 实时目标检测 / 实例分割。

    优先使用 roboflow `rfdetr` pip 包（RFDETRBase / RFDETRSegmentation）；
    未安装时回退到 transformers 的 DETR 系加载（trust_remote_code，仅目标检测）。
    分割模型（rf-detr-segmentation）额外返回 masks。

    TODO：rfdetr 包的构造参数与 Detection 对象属性（boxes/masks）按官方 README 校准。
    """

    MODELS = (
        "Roboflow/rf-detr",
        "Roboflow/rf-detr-segmentation",
    )

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
            self._processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.model_id,
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


@register_adapter(category="detection")
class GroundingDINOAdapter(BaseAdapter):
    """IDEA-Research grounding-dino 开放词表目标检测。

    使用 groundingdino 官方仓库推理（groundingdino.util.inference），
    文本提示词通过 inputs.prompt 或 params.prompt 传入。
    权重与配置从 HF 仓库快照中定位后交给官方 load_model。

    TODO：config/权重文件定位规则与 predict 的图像格式（RGB/BGR）
    按 grounding-dino-1.5-pro 官方 README 校准后上线。
    """

    MODELS = ("IDEA-Research/grounding-dino-1.5-pro",)

    def _load(self) -> None:
        import glob
        import os

        import torch
        from groundingdino.util.inference import load_model
        from huggingface_hub import snapshot_download

        self._device_obj = torch.device(self.device)
        repo_dir = snapshot_download(repo_id=self.model_id)
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


@register_adapter(category="detection")
class EdgeTAMAdapter(BaseAdapter):
    """facebook/EdgeTAM 实时视频分割（边缘设备部署）。

    官方推理代码待接入 — see TODO。
    待接入内容：EdgeTAM 官方 checkpoint 与推理脚本
    （单帧 prompt 交互分割 / 视频流帧间传播分割）。
    """

    MODELS = ("facebook/EdgeTAM",)

    def _load(self) -> None:
        raise NotImplementedError("facebook/EdgeTAM 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("facebook/EdgeTAM 官方推理代码待接入 — see TODO")


@register_adapter(category="detection")
class DEIMAdapter(BaseAdapter):
    """Intellindust/DEIMv2_DINOv3_X_COCO 实时目标检测。

    官方推理代码待接入 — see TODO。
    待接入内容：DEIM 官方 DETR 推理脚本（加载 config + 权重，
    非极大值抑制与标签映射按官方 README 校准）。
    """

    MODELS = ("Intellindust/DEIMv2_DINOv3_X_COCO",)

    def _load(self) -> None:
        raise NotImplementedError("Intellindust/DEIMv2 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Intellindust/DEIMv2 官方推理代码待接入 — see TODO")
