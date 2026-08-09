# ============================================================================
# 语音合成（TTS）适配器（13 个模型）
#
# 适配器族：
#   1. TransformersTTSAdapter —— transformers AutoModelForTextToSpeech（Qwen3-TTS）
#   2. CosyVoiceAdapter       —— FunASR 加载 CosyVoice3（零样本克隆/多说话人）
#   3. 厂商专用适配器         —— fish-speech / zonos / voxcpm / sesame_csm
#                                / orpheus-tts / index_tts（各 ~40-70 行，尽力实现）
#   4. TODO 骨架适配器        —— MOSS-TTS / VibeVoice / OmniVoice / neutts-2e
#                                （依赖厂商官方运行时，暂以 NotImplementedError 占位）
#
# /predict 输入输出约定：
#   {"text": str, "speaker": str|None}
#     -> {"audio": "<base64 wav>", "sample_rate": int, "format": "wav"}
#   - speaker 语义随模型而定（说话人 ID / 音色描述 / 参考音频 base64），
#     由各适配器按官方 README 解释。
# ============================================================================

import os
import tempfile
from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter
from ..utils.io_codec import audio_to_b64, b64_to_bytes, numpy_audio_to_b64


@register_adapter(category="tts")
class TransformersTTSAdapter(BaseAdapter):
    """transformers 原生 TTS：AutoModel 加载 Qwen3-TTS（qwen3_tts 架构）。

    支持通过 params.speaker_description 传入音色描述（CustomVoice 特性）。
    注：需要 transformers>=5.15（per-model 依赖覆盖，2026-08 实测 4.57/5.14 均不支持）。
    """

    MODELS = ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",)

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device_obj = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._load_ref)
        self._model = AutoModel.from_pretrained(
            self._load_ref,
            torch_dtype=torch.float16 if self._device_obj.type == "cuda" else torch.float32,
        )
        self._model.eval().to(self._device_obj)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        tokens = self._tokenizer(text, return_tensors="pt").to(self._device_obj)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.get("max_new_tokens", self.config.get("max_new_tokens", 2048))),
        }
        if speaker:
            # 音色描述（CustomVoice 支持自然语言描述说话人音色）
            gen_kwargs["speaker_description"] = speaker

        with torch.inference_mode():
            output = self._model.generate(**tokens, **gen_kwargs)

        waveform = output.audio.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = int(getattr(output, "sample_rate", 24000))
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class CosyVoiceAdapter(BaseAdapter):
    """FunASR 加载 CosyVoice3（Fun-AudioLLM）语音合成。

    零样本克隆：传入 inputs.prompt_speech（参考音频 base64）与可选
    params.prompt_text（参考文本）即可克隆音色。
    """

    MODELS = ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512",)

    def _load(self) -> None:
        # 延迟导入：funasr 依赖在 requirements/tts.txt 中按需安装
        from funasr import AutoModel

        self._model = AutoModel(
            model=self.model_id,
            trust_remote_code=True,
            device=self.device if self.device.startswith("cuda") else "cpu",
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        gen_kwargs: dict[str, Any] = {
            "text": text,
            "spk_id": str(inputs.get("speaker") or params.get("speaker") or "中文女"),
            "stream": False,
        }
        prompt_text = inputs.get("prompt_text") or params.get("prompt_text")
        if prompt_text:
            gen_kwargs["prompt_text"] = prompt_text

        prompt_path = None
        prompt_b64 = inputs.get("prompt_speech") or inputs.get("prompt_audio")
        if prompt_b64:
            # CosyVoice 支持 prompt_speech 参考音色；funasr 接受 wav 文件路径
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b64_to_bytes(prompt_b64))
                prompt_path = tmp.name
            gen_kwargs["prompt_speech"] = prompt_path
        try:
            result = self._model.generate(**gen_kwargs)
        finally:
            if prompt_path:
                os.unlink(prompt_path)

        # 兼容生成器返回；离线（stream=False）通常返回 list[dict]
        if hasattr(result, "__next__"):
            result = list(result)
        # TODO: 按官方 README 校准字段名；CosyVoice3 的 result 结构可能与 v2 略有差异
        audio = result[0]["audio"]  # np.ndarray [T]
        sr = int(result[0].get("sample_rate", 24000))
        return {"audio": numpy_audio_to_b64(audio, sr), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class FishSpeechAdapter(BaseAdapter):
    """fish-speech 语音合成（fishaudio/s2-pro）。

    官方仓库：https://github.com/fishaudio/fish-speech
    离线推理依赖该仓库的 config + checkpoint（.yaml/.ckpt），
    此处按官方推荐接入本地推理服务（OpenAI 兼容 API）。
    """

    MODELS = ("fishaudio/s2-pro",)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；s2-pro 仓库含 v1/v2 双版本，需按 config 指定
        from fish_speech.api import API  # 厂商客户端（在线/本地推理服务）

        self._api = API(
            model="s2-pro",
            base_url=self.config.get("api_base", "http://127.0.0.1:8000/v1"),
            api_key=self.config.get("api_key", ""),
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import base64

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        # TODO: 按官方 README 校准返回类型（bytes / PlaybackResponse / numpy）
        response = self._api.tts(text, voice=speaker or "default", format="wav")
        if hasattr(response, "audio"):
            audio_bytes = response.audio
        else:
            audio_bytes = bytes(response)
        sr = int(params.get("sample_rate", 44100))  # fish-speech 默认 44100
        return {
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": sr,
            "format": "wav",
        }


@register_adapter(category="tts")
class ZonosAdapter(BaseAdapter):
    """Zyphra Zonos 语音合成（ZonosPipeline）。

    官方仓库：https://github.com/Zyphra/Zonos
    注：Zonos 需要 speaker embedding / emotion 等条件输入，按 README 校准。
    """

    MODELS = ("Zyphra/Zonos",)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；优先 ZonosPipeline，回退到 Zonos 类
        try:
            from zonos import ZonosPipeline

            self._model = ZonosPipeline.from_pretrained(self.model_id, device=self.device)
        except ImportError:
            from zonos import Zonos

            self._model = Zonos.from_pretrained(self.model_id, device=self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        gen_kwargs: dict[str, Any] = {}
        if speaker:
            # Zonos 支持 speaker description 作为条件输入
            gen_kwargs["speaker_description"] = speaker

        with torch.inference_mode():
            # TODO: synthesize 方法名与返回结构按官方 README 校准
            output = self._model.synthesize(text, **gen_kwargs)

        if isinstance(output, tuple):
            waveform, sr = output
        else:
            waveform, sr = output, int(params.get("sample_rate", 24000))
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class VoxCPMAdapter(BaseAdapter):
    """OpenBMB VoxCPM2 语音合成（voxcpm 包）。

    官方仓库：https://github.com/OpenBMB/VoxCPM
    """

    MODELS = ("openbmb/VoxCPM2",)

    def _load(self) -> None:
        import torch
        # TODO: 按官方 README 校准加载方式（VoxCPMConfig + VoxCPM）
        from voxcpm import VoxCPM, VoxCPMConfig

        config = VoxCPMConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = VoxCPM.from_pretrained(
            self.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        )
        self._model.eval().to(self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        speaker_id = int(speaker) if str(speaker).isdigit() else 0

        with torch.inference_mode():
            # TODO: VoxCPM 使用 fixed_prompt / mm_token 等输入，按官方 README 校准
            output = self._model.generate(text=text, speaker_id=speaker_id, max_new_tokens=2048)

        if isinstance(output, tuple):
            waveform, sr = output
        else:
            waveform, sr = output, int(params.get("sample_rate", 16000))
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class CSMAdapter(BaseAdapter):
    """Sesame CSM-1B 语音合成（sesame_csm 包）。

    官方仓库：https://github.com/SesameAILabs/csm
    """

    MODELS = ("sesame/csm-1b",)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；load_csm_1b 直接加载原生权重
        from sesame_csm.generator import load_csm_1b

        self._generator = load_csm_1b(self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        try:
            speaker_id = int(speaker) if speaker else 0
        except (TypeError, ValueError):
            speaker_id = 0

        with torch.inference_mode():
            # TODO: CSM 支持 context（多轮对话）输入，按官方 README 校准
            audio = self._generator.generate(
                text=text, speaker=speaker_id, max_audio_length_ms=int(params.get("max_audio_length_ms", 30000))
            )

        # 返回 numpy int16/float32 波形 -> 转 torch.Tensor 后编码
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        sr = 24000
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class OrpheusAdapter(BaseAdapter):
    """Canopy Labs Orpheus 语音合成（orpheus-tts 包）。

    官方仓库：https://github.com/canopyai/Orpheus-TTS
    """

    MODELS = ("canopylabs/orpheus-3b-0.1-ft",)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；orpheus-tts 提供 Orpheus 推理类
        from orpheus_tts import Orpheus

        self._model = Orpheus(model_path=self.model_id, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        emotion = params.get("emotion", "None")  # Orpheus 支持 tara / leah 等情绪

        # TODO: Orpheus 先产出音素 token 序列，再经 tokenizer 解码为音频，按官方 README 校准
        syn_tokens = self._model.generate_speech(text, emotion=emotion, voice=speaker or "tara")
        waveform = self._model.tokenizer.decode(syn_tokens[0])
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        sr = 24000
        return {"audio": audio_to_b64(waveform, sr, fmt="wav"), "sample_rate": sr, "format": "wav"}


@register_adapter(category="tts")
class IndexTTSAdapter(BaseAdapter):
    """IndexTeam IndexTTS2 语音合成（index_tts 包）。

    官方仓库：https://github.com/index-tts/index-tts
    speaker 输入可接受参考音频 base64（音色克隆）或说话人 ID。
    """

    MODELS = ("IndexTeam/IndexTTS2",)

    def _load(self) -> None:
        # TODO: 按官方 README 校准；index_tts 提供 IndexTTS2 类
        from index_tts import IndexTTS2

        self._tts = IndexTTS2(device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not text or not isinstance(text, str):
            raise InvalidInputError("inputs 必须包含 'text'（str）")

        speaker = inputs.get("speaker") or params.get("speaker")
        ref_path = None
        if speaker:
            # 参考音频 base64 -> 临时 wav；TODO: 若 speaker 为说话人 ID 则跳过此步
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(b64_to_bytes(speaker))
                ref_path = tmp.name
        try:
            # TODO: tts 方法与返回结构（np.ndarray, sr）按官方 README 校准
            audio_np, sr = self._tts.tts(text, ref_audio_path=ref_path)
        finally:
            if ref_path:
                os.unlink(ref_path)
        return {"audio": numpy_audio_to_b64(audio_np, int(sr)), "sample_rate": int(sr), "format": "wav"}


class _TODOTTSAdapter(BaseAdapter):
    """TTS 骨架适配器基类：依赖厂商官方运行时，尚未接入。"""

    #: 厂商/运行时名称，用于异常信息
    _vendor = "unknown"

    def _load(self) -> None:
        raise NotImplementedError(
            f"requires official {self._vendor} runtime — see TODO（尚未接入 {self._vendor} 官方运行时）"
        )

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(
            f"requires official {self._vendor} runtime — see TODO（尚未接入 {self._vendor} 官方运行时）"
        )


@register_adapter(category="tts")
class MOSSTTSAdapter(_TODOTTSAdapter):
    """OpenMOSS-Team/MOSS-TTS：MOSS 官方语音合成运行时（骨架占位）。"""

    MODELS = ("OpenMOSS-Team/MOSS-TTS",)
    _vendor = "MOSS-TTS"


@register_adapter(category="tts")
class VibeVoiceAdapter(_TODOTTSAdapter):
    """microsoft/VibeVoice：微软实时语音合成运行时（骨架占位）。"""

    MODELS = (
        "microsoft/VibeVoice-Realtime-0.5B",
        "microsoft/VibeVoice-1.5B",
    )
    _vendor = "VibeVoice"


@register_adapter(category="tts")
class OmniVoiceAdapter(_TODOTTSAdapter):
    """k2-fsa/OmniVoice：k2-fsa 官方语音合成运行时（骨架占位）。"""

    MODELS = ("k2-fsa/OmniVoice",)
    _vendor = "OmniVoice"


@register_adapter(category="tts")
class NeuttsAdapter(_TODOTTSAdapter):
    """neuphonic/neutts-2e：neuphonic 官方语音合成运行时（骨架占位）。"""

    MODELS = ("neuphonic/neutts-2e",)
    _vendor = "neuphonic"
