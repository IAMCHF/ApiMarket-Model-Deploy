"""
numind/NuMarkdown-8B-Thinking 专用适配器（自动生成，请勿手工编辑）。
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


class VLMOCRAdapter(BaseAdapter):
    """transformers 多模态 VLM：图片 + 提示词 -> 生成文本/结构化输出。

    模型间差异通过 PROMPTS 表（系统提示词）与 trust_remote_code 控制；
    个别模型（olmOCR / MinerU / dots.ocr）官方提供专用加载脚本，
    上线前按官方 README 校准 processor 与 generate 参数。
    """

    # 模型 -> 默认提示词（可被请求 params.prompt 覆盖）
    PROMPTS: dict[str, str] = {
        "baidu/Unlimited-OCR": "识别图片中的全部文字，输出为 Markdown 格式，保留表格结构。",
        "ATH-MaaS/OvisOCR2": "OCR 识别图片中所有文字，保留原始排版，输出 Markdown。",
        "deepseek-ai/DeepSeek-OCR-2": "提取图片中的全部文字与表格，输出结构化 Markdown。",
        "tencent/HunyuanOCR": "识别图片文字，复杂表格与手写体也要识别，输出 Markdown。",
        "numind/NuMarkdown-8B-Thinking": "将图片转换为 Markdown，保留标题/列表/表格等排版。",
        "lightonai/LightOnOCR-2-1B": "多语言 OCR，提取图片全部文字输出为 Markdown。",
        "nvidia/NVIDIA-Nemotron-Parse-v1.2": "解析文档图片，提取结构化信息，输出 Markdown。",
        "openbmb/MiniCPM-V-4.6": "识别图片中的文字内容，输出 Markdown。",
        "microsoft/Fara1.5-9B": "文档理解与 OCR，输出图片中的文字为 Markdown。",
        "microsoft/Fara1.5-4B": "文档理解与 OCR，输出图片中的文字为 Markdown。",
        "rednote-hilab/dots.ocr": "版面解析：将文档图片转为 Markdown，保留阅读顺序。",
        "allenai/olmOCR-2-7B-1025": "将扫描件/PDF 页转写为干净 Markdown，保留阅读顺序。",
        "opendatalab/MinerU2.5-Pro-2604-1.2B": "高分辨率文档解析：输出 Markdown 与结构化 JSON。",
        "Qwen/Qwen3-VL-8B-Instruct": "读取图片中的文字，输出 Markdown。",
        "OpenGVLab/InternVL3-2B-Instruct": "读取图片中的文字与图表内容，输出 Markdown。",
        "THUDM/GLM-4.1V-9B-Thinking": "读取图片中的文字，输出 Markdown。",
        "HuggingFaceTB/SmolVLM-Instruct": "读取截图/文档中的文字并回答，输出 Markdown。",
        "zai-org/GLM-OCR": "识别复杂文档中的文字，输出 Markdown。",
    }

    MODELS = tuple(PROMPTS.keys())

    def _load(self) -> None:
        import torch

        self._device_obj = torch.device(self.device)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))

        if self.model_id == "baidu/Unlimited-OCR":
            # 官方推理路径：AutoModel + AutoTokenizer + model.infer（见官方 README）
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
            self._model = AutoModel.from_pretrained(self._load_ref,
                trust_remote_code=trust_remote_code,
                use_safetensors=True,
                torch_dtype=torch.bfloat16 if self._device_obj.type == "cuda" else torch.float32,
            )
            self._model.eval().to(self._device_obj)
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._load_ref, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(self._load_ref,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    @staticmethod
    def _read_output_dir(output_dir: str) -> str:
        """扫描推理输出目录，合并全部文本结果（result.md / result.txt / result.json 等）。

        Unlimited-OCR 的 infer 输出 result.md（OCR 文本）与 result_with_boxes.jpg（可视化）。
        """
        import glob
        import os

        texts = []
        for path in sorted(glob.glob(os.path.join(output_dir, "**", "*"), recursive=True)):
            if not os.path.isfile(path):
                continue
            if path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
            except OSError:
                continue
            if content:
                texts.append(content)
        return "\n".join(texts)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        image_b64 = inputs.get("image") or inputs.get("images")
        if not image_b64:
            raise InvalidInputError("inputs must contain 'image' (base64 str)")
        image = b64_to_image(image_b64)

        prompt = params.get("prompt") or self.PROMPTS.get(self.model_id, "识别图片中的全部文字。")
        max_new_tokens = int(params.get("max_new_tokens", self.config.get("max_new_tokens", 2048)))

        # ---- baidu/Unlimited-OCR：官方 infer 接口（输出写入临时目录） ----
        if self.model_id == "baidu/Unlimited-OCR":
            import os
            import shutil
            import tempfile

            from server import b64_to_bytes

            infer_prompt = params.get("prompt") or "<image>document parsing."
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(b64_to_bytes(image_b64))
                image_path = tmp.name
            output_dir = tempfile.mkdtemp(prefix="unlimitedocr_out_")
            try:
                result = self._model.infer(
                    self._tokenizer,
                    prompt=infer_prompt,
                    image_file=image_path,
                    output_path=output_dir,
                    base_size=int(params.get("base_size", 1024)),
                    image_size=int(params.get("image_size", 640)),
                    crop_mode=bool(params.get("crop_mode", True)),
                    max_length=int(params.get("max_new_tokens", 32768)),
                    no_repeat_ngram_size=int(params.get("no_repeat_ngram_size", 35)),
                    ngram_window=int(params.get("ngram_window", 128)),
                    save_results=True,
                )
            finally:
                os.unlink(image_path)
            text = ""
            if isinstance(result, str) and result.strip():
                text = result
            else:
                text = self._read_output_dir(output_dir)
            shutil.rmtree(output_dir, ignore_errors=True)
            return {"text": text}

        # ---- 通用 VLM 路径 ----
        # 多数 VLM 使用 chat 模板；不支持的模型退化为纯文本 prompt
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text_input = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, TypeError):
            text_input = prompt

        batch = self._processor(text=text_input, images=image, return_tensors="pt").to(self._device_obj)
        with torch.inference_mode():
            generated = self._model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False
            )
        # 去掉输入 token，仅保留生成文本
        prompt_len = batch["input_ids"].shape[1]
        output_ids = generated[:, prompt_len:]
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return {"text": text}

    category = 'ocr'



# 统一适配器入口（server.py 加载此名称）
Adapter = VLMOCRAdapter
