"""
Wan-AI/Wan2.1-T2V-1.3B 专用适配器（自动生成，请勿手工编辑）。
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


class DiffusersVideoAdapter(BaseAdapter):
    """diffusers 视频生成（Wan-AI/Wan2.1-T2V-1.3B、tencent/HunyuanVideo-1.5）。

    按 model_id 选择对应 diffusers 管线：Wan2.1 -> WanPipeline，
    HunyuanVideo -> HunyuanVideoPipeline。生成帧经 imageio-ffmpeg
    （imageio FFMPEG 插件）导出为临时 MP4 后以 base64 返回。

    TODO：输出视频导出方案（fps / 首帧拼接 / 帧循环）按各模型官方
    README 校准后上线。
    """

    MODELS = ('Wan-AI/Wan2.1-T2V-1.3B',)

    # 模型级默认生成参数（可被请求 params 覆盖）
    DEFAULTS: dict[str, dict[str, Any]] = {
        "Wan-AI/Wan2.1-T2V-1.3B": {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "num_inference_steps": 30,
            "fps": 16,
        },
        "tencent/HunyuanVideo-1.5": {
            "height": 480,
            "width": 832,
            "num_frames": 129,
            "num_inference_steps": 30,
            "fps": 24,
        },
    }

    def _load(self) -> None:
        import torch
        from diffusers import HunyuanVideoPipeline, WanPipeline

        self._device_obj = torch.device(self.device)
        # Wan2.1 使用 WanPipeline；HunyuanVideo 使用 HunyuanVideoPipeline
        pipeline_cls = WanPipeline if "Wan" in self.model_id else HunyuanVideoPipeline
        self._pipeline = pipeline_cls.from_pretrained(self._load_ref,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
        )
        self._pipeline.to(self._device_obj)
        self._fps = int(self.DEFAULTS.get(self.model_id, {}).get("fps", 16))

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import os
        import tempfile

        import numpy as np
        import torch

        prompt = inputs.get("prompt")
        if not prompt:
            raise InvalidInputError("inputs 必须包含 'prompt'（文本提示词）")

        defaults = self.DEFAULTS.get(self.model_id, {})
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": inputs.get("negative_prompt", params.get("negative_prompt", "")),
            "height": int(params.get("height", defaults.get("height", 480))),
            "width": int(params.get("width", defaults.get("width", 832))),
            "num_frames": int(params.get("num_frames", defaults.get("num_frames", 81))),
            "num_inference_steps": int(params.get("num_inference_steps", defaults.get("num_inference_steps", 30))),
        }
        if params.get("guidance_scale") is not None:
            gen_kwargs["guidance_scale"] = float(params["guidance_scale"])
        if params.get("seed") is not None:
            torch.manual_seed(int(params["seed"]))

        with torch.inference_mode():
            output = self._pipeline(**gen_kwargs)

        # 管线返回 .frames：List[PIL.Image] 或 List[List[PIL.Image]]（外层 batch）
        frames = output.frames
        if frames and isinstance(frames[0], (list, tuple)):
            frames = frames[0]
        if not frames:
            raise RuntimeError("视频管线未生成任何帧")

        fps = int(params.get("fps", self._fps))

        # 经 imageio-ffmpeg（imageio FFMPEG 插件）将帧序列导出为 MP4
        import imageio.v2 as imageio

        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            writer = imageio.get_writer(tmp_path, fps=fps, format="FFMPEG")
            for frame in frames:
                writer.append_data(np.asarray(frame))
            writer.close()
            with open(tmp_path, "rb") as fh:
                video_b64 = bytes_to_b64(fh.read())
        finally:
            os.unlink(tmp_path)

        return {"video": video_b64, "format": "mp4", "fps": fps}

    category = 'video-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = DiffusersVideoAdapter
