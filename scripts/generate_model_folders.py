#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# generate_model_folders.py — 为 100 个模型生成完全隔离的部署文件夹
#
# 输出：models/<组织名-模型名>/
#   server.py        自包含服务框架（BaseAdapter + io_codec + FastAPI，固定 0.0.0.0:8080）
#   adapter.py       该模型专用适配器（从 app/adapters 提取并单模型化）
#   requirements.txt 依赖清单（common + 类别）
#   start.sh         统一启动脚本（Linux，含换源步骤）
#   start.bat        统一启动脚本（Windows，含换源步骤）
#   weights/         权重放置目录（用户预置，自动保留不覆盖）
#
# 用法：python scripts/generate_model_folders.py
# 说明：
#   - 重复运行会覆盖代码文件，但不会删除 weights/ 中已放置的权重
#   - 所有文件强制 LF 行尾（Windows 下 write_text 默认 CRLF 会破坏容器内脚本）
# ============================================================================

import argparse
import ast
import inspect
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_OUTPUT = REPO_ROOT / "models"
TEST_OUTPUT = REPO_ROOT / "test"  # 测试脚本根目录：test/test-<模型名>/test-<模型名>.py

# 加载引用替换的关键字参数名（from_pretrained 类调用的第一个位置参数亦替换）
_LOAD_KWARGS = (
    "model",
    "model_id",
    "model_name",
    "model_path",
    "repo_id",
    "pretrained_path",
    "pretrained_model_name_or_path",
)

# 模型级附加依赖：类别级依赖存在冲突时（如不同模型要求不同 transformers 版本），
# 在生成 requirements.txt 时追加到该模型文件夹，保证"装完即启动"。
# 注意：pip 后装者覆盖先装者，故覆盖版本（如 transformers 降级）放在此处。
PER_MODEL_REQUIREMENTS: dict[str, list[str]] = {
    "baidu/Unlimited-OCR": [
        # 官方 README 指定 transformers 4.57.1；镜像 5.x 移除了 modeling 所需的
        # transformers.utils.import_utils.is_torch_fx_available（2026-08 实测）
        "transformers==4.57.1",
        "matplotlib>=3.8.0",
    ],
}

# ============================================================================
# 测试脚本：gen_inputs() 按适配器类生成测试输入（可被用户手工扩展）
# ============================================================================
# 测试脚本 gen_inputs() 按适配器类构造 inputs（数据来自 test/<模型>/data/ 预置文件）
# 测试脚本 gen_inputs() 按适配器类构造 inputs（数据来自 test/<模型>/data/ 预置文件）
_TEST_GEN: dict[str, str] = {
    "SentenceTransformerAdapter": "return {'texts': _load_json('texts.json')['texts']}",
    "HuggingFaceEmbeddingAdapter": "return {'texts': _load_json('texts.json')['texts']}",
    "RerankerAdapter": "return {'query': _load_text('query.txt'), 'documents': _load_json('documents.json')}",
    "VLMOCRAdapter": "return {'image': _load_b64('image.png')}",
    "PaddleOCRAdapter": "return {'image': _load_b64('image.png')}",
    "DoclingAdapter": "return {'image': _load_b64('image.png')}",
    "GOTOCRAdapter": "return {'image': _load_b64('image.png')}",
    "NuExtractAdapter": "return {'document': _load_text('document.txt'), 'schema': _load_json('schema.json')}",
    "NeMoASRAdapter": "return {'audio': _load_b64('audio.wav')}",
    "PyannoteDiarizationAdapter": "return {'audio': _load_b64('audio.wav')}",
    "HuggingFaceASRAdapter": "return {'audio': _load_b64('audio.wav')}",
    "MOSSASRDiarizeAdapter": "return {'audio': _load_b64('audio.wav')}",
    "DiffusersT2IAdapter": "return {'prompt': _load_text('prompt.txt')}",
    "DiffusersEditAdapter": "return {'prompt': _load_text('prompt.txt'), 'image': _load_b64('image.png')}",
    "DiffusersVideoAdapter": "return {'prompt': _load_text('prompt.txt')}",
    "MobileWanAdapter": "return {'prompt': _load_text('prompt.txt')}",
    "TRELLISAdapter": "return {'image': _load_b64('image.png')}",
    "Hunyuan3DAdapter": "return {'image': _load_b64('image.png')}",
    "StableFast3DAdapter": "return {'image': _load_b64('image.png')}",
    "TripoSGAdapter": "return {'image': _load_b64('image.png')}",
    "InstantNuRecAdapter": "return {'image': _load_b64('image.png')}",
    "Step1X3DAdapter": "return {'image': _load_b64('image.png')}",
    "SAM3Adapter": "return {'image': _load_b64('image.png')}",
    "RMBGAdapter": "return {'image': _load_b64('image.png')}",
    "RFDetrAdapter": "return {'image': _load_b64('image.png')}",
    "GroundingDINOAdapter": "return {'image': _load_b64('image.png'), 'prompt': _load_text('prompt.txt')}",
    "EdgeTAMAdapter": "return {'image': _load_b64('image.png')}",
    "DEIMAdapter": "return {'image': _load_b64('image.png')}",
    "GR00TAdapter": "return {'image': _load_b64('image.png'), 'instruction': _load_text('instruction.txt')}",
    "LlamaCppCodeGenAdapter": "return {'prompt': _load_text('prompt_code.txt')}",
    "TransformersCodeGenAdapter": "return {'prompt': _load_text('prompt_code.txt')}",
    "ChronosAdapter": "return _load_json('forecast.json')",
    "TimesFMAdapter": "return _load_json('forecast.json')",
    "PrivacyFilterAdapter": "return {'text': _load_text('text.txt')}",
    "AntaresAdapter": "return {'code': _load_text('code.txt')}",
    "TransformersTTSAdapter": "return {'text': _load_text('text.txt')}",
    "CosyVoiceAdapter": "return {'text': _load_text('text.txt')}",
    "FishSpeechAdapter": "return {'text': _load_text('text.txt')}",
    "ZonosAdapter": "return {'text': _load_text('text.txt')}",
    "VoxCPMAdapter": "return {'text': _load_text('text.txt')}",
    "CSMAdapter": "return {'text': _load_text('text.txt')}",
    "OrpheusAdapter": "return {'text': _load_text('text.txt')}",
    "IndexTTSAdapter": "return {'text': _load_text('text.txt')}",
    "MOSSTTSAdapter": "return {'text': _load_text('text.txt')}",
    "VibeVoiceAdapter": "return {'text': _load_text('text.txt')}",
    "OmniVoiceAdapter": "return {'text': _load_text('text.txt')}",
    "NeuttsAdapter": "return {'text': _load_text('text.txt')}",
}


# 测试数据文件清单：适配器类 -> 需要的 data/ 文件
_TEST_DATA_FILES: dict[str, list[str]] = {
    "SentenceTransformerAdapter": ["texts.json"],
    "HuggingFaceEmbeddingAdapter": ["texts.json"],
    "RerankerAdapter": ["query.txt", "documents.json"],
    "VLMOCRAdapter": ["image.png"],
    "PaddleOCRAdapter": ["image.png"],
    "DoclingAdapter": ["image.png"],
    "GOTOCRAdapter": ["image.png"],
    "NuExtractAdapter": ["document.txt", "schema.json"],
    "NeMoASRAdapter": ["audio.wav"],
    "PyannoteDiarizationAdapter": ["audio.wav"],
    "HuggingFaceASRAdapter": ["audio.wav"],
    "MOSSASRDiarizeAdapter": ["audio.wav"],
    "TransformersTTSAdapter": ["text.txt"],
    "CosyVoiceAdapter": ["text.txt"],
    "FishSpeechAdapter": ["text.txt"],
    "ZonosAdapter": ["text.txt"],
    "VoxCPMAdapter": ["text.txt"],
    "CSMAdapter": ["text.txt"],
    "OrpheusAdapter": ["text.txt"],
    "IndexTTSAdapter": ["text.txt"],
    "MOSSTTSAdapter": ["text.txt"],
    "VibeVoiceAdapter": ["text.txt"],
    "OmniVoiceAdapter": ["text.txt"],
    "NeuttsAdapter": ["text.txt"],
    "DiffusersT2IAdapter": ["prompt.txt"],
    "DiffusersEditAdapter": ["prompt.txt", "image.png"],
    "DiffusersVideoAdapter": ["prompt.txt"],
    "MobileWanAdapter": ["prompt.txt"],
    "TRELLISAdapter": ["image.png"],
    "Hunyuan3DAdapter": ["image.png"],
    "StableFast3DAdapter": ["image.png"],
    "TripoSGAdapter": ["image.png"],
    "InstantNuRecAdapter": ["image.png"],
    "Step1X3DAdapter": ["image.png"],
    "SAM3Adapter": ["image.png"],
    "RMBGAdapter": ["image.png"],
    "RFDetrAdapter": ["image.png"],
    "GroundingDINOAdapter": ["image.png", "prompt.txt"],
    "EdgeTAMAdapter": ["image.png"],
    "DEIMAdapter": ["image.png"],
    "GR00TAdapter": ["image.png", "instruction.txt"],
    "LlamaCppCodeGenAdapter": ["prompt_code.txt"],
    "TransformersCodeGenAdapter": ["prompt_code.txt"],
    "ChronosAdapter": ["forecast.json"],
    "TimesFMAdapter": ["forecast.json"],
    "PrivacyFilterAdapter": ["text.txt"],
    "AntaresAdapter": ["code.txt"],
}


def _test_data_content(name: str) -> bytes:
    """生成测试数据文件内容（预置到 test/<模型>/data/，脚本运行时只读取）。"""
    import io

    if name == "image.png":
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 180), "white")
        ImageDraw.Draw(img).text((30, 60), "TEST 123", fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    if name == "audio.wav":
        import math
        import struct
        import wave

        sr, dur, freq = 16000, 1.5, 440
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"".join(
                struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * t / sr)))
                for t in range(int(dur * sr))
            ))
        return buf.getvalue()
    if name == "texts.json":
        return json.dumps({"texts": ["今天天气很好", "模型部署测试"]}, ensure_ascii=False).encode()
    if name == "query.txt":
        return "如何部署大模型".encode()
    if name == "documents.json":
        return json.dumps({"documents": ["模型部署教程", "今天吃什么"]}, ensure_ascii=False).encode()
    if name == "text.txt":
        return "你好，欢迎使用语音合成服务。".encode()
    if name == "prompt.txt":
        return "a cute cat, high quality".encode()
    if name == "prompt_code.txt":
        return "def add(a, b):\n    return a + b".encode()
    if name == "document.txt":
        return "甲方于2026年8月采购服务器20台，金额100万元。".encode()
    if name == "schema.json":
        return json.dumps({"购买方": "", "数量": "", "金额": ""}, ensure_ascii=False).encode()
    if name == "forecast.json":
        return json.dumps({
            "context": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "prediction_length": 5,
        }).encode()
    if name == "instruction.txt":
        return "pick up the object".encode()
    if name == "code.txt":
        return "def auth(request):\n    return request.user".encode()
    raise ValueError(f"未定义测试数据文件: {name}")


def _write_lf(path: Path, text: str) -> None:
    """写文件并强制 LF 行尾（Windows 下避免 CRLF 破坏容器内 bash 脚本）。"""
    path.write_text(text, encoding="utf-8", newline="\n")


def _gen_inputs_for(cls_name: str) -> str:
    return _TEST_GEN.get(cls_name, "return {'text': 'hello world'}")


def build_test_script(model_id: str, cls_name: str) -> str:
    """生成 test-<模型名>.py：读模板并填充模型 ID / gen_inputs。"""
    tpl = (TEMPLATES_DIR / "test_script.py.tpl").read_text(encoding="utf-8")
    return (
        tpl.replace("__MODEL_ID__", model_id)
        .replace("__TEST_NAME__", f"test-{model_id.replace('/', '-')}")
        .replace("__GEN_INPUTS__", _gen_inputs_for(cls_name))
    )


def build_test_data(cls_name: str) -> dict[str, bytes]:
    """生成该适配器类所需的测试数据文件（内容）。"""
    return {name: _test_data_content(name) for name in _TEST_DATA_FILES.get(cls_name, [])}


def _rewrite_io_imports(text: str) -> str:
    """将相对导入改写为指向同目录 server.py（自包含隔离）。"""
    text = re.sub(r"from \.\.utils\.io_codec import", "from server import", text)
    text = re.sub(r"from \.base import", "from server import", text)
    text = re.sub(r"from \.registry import[^\n]*", "", text)
    return text


def _extract_category(cls_node: ast.ClassDef) -> str | None:
    """从装饰器 @register_adapter(category=...) 提取类别（源码段不含装饰器，须走 AST）。"""
    for dec in cls_node.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "id", None) == "register_adapter":
            for kw in dec.keywords:
                if kw.arg == "category" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def _transform_class(source: str, cls_node: ast.ClassDef, model_id: str) -> str:
    """对目标类源码做单模型化变换（保留注释）。"""
    text = ast.get_source_segment(source, cls_node)
    category = _extract_category(cls_node)
    text = re.sub(r"^@register_adapter\([^\n]*\)\n", "", text, flags=re.M)
    text = re.sub(
        r"MODELS\s*=\s*\([^)]*\)",
        f"MODELS = ({model_id!r},)",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"from_pretrained\(\s*self\.model_id",
        "from_pretrained(self._load_ref",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"SentenceTransformer\(\s*self\.model_id",
        "SentenceTransformer(self._load_ref",
        text,
    )
    kw_alt = "|".join(_LOAD_KWARGS)
    text = re.sub(
        rf"\b({kw_alt})=self\.model_id\b",
        r"\1=self._load_ref",
        text,
    )
    text = _rewrite_io_imports(text)
    if category:
        text = text.rstrip() + f"\n\n    category = {category!r}\n"
    return text


def _collect_dependencies(tree: ast.Module, target: str, source: str) -> list[str]:
    """收集目标类依赖的模块级函数/类（基类 mixin、helper 函数），按源码顺序返回。"""
    module_defs = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    needed = {target}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in needed:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Name)
                        and isinstance(sub.ctx, ast.Load)
                        and sub.id in module_defs
                        and sub.id not in needed
                    ):
                        needed.add(sub.id)
                        changed = True
    segments = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in needed and node.name != target:
            segments.append(_rewrite_io_imports(ast.get_source_segment(source, node)))
    return segments


def _special_patches(model_id: str, class_name: str, text: str) -> str:
    """对无法通用替换的加载逻辑做定向补丁（指向本地 weights/）。"""
    if class_name == "LlamaCppCodeGenAdapter":
        text = text.replace(
            'models_root = os.environ.get("MODELS_ROOT", "/data/models")\n        model_dir = os.path.join(models_root, self.model_id)',
            'model_dir = self._weights_dir if self._weights_dir else os.path.join(os.environ.get("MODELS_ROOT", "/data/models"), self.model_id)',
        )
        text = text.replace("tools/download_models.py", "weights/ 权重预置")
    elif class_name == "GroundingDINOAdapter":
        text = text.replace(
            "repo_dir = snapshot_download(repo_id=self._load_ref)",
            "repo_dir = self._weights_dir if self._weights_dir else snapshot_download(repo_id=self.model_id)",
        )
    elif class_name == "GR00TAdapter":
        text = text.replace(
            'models_root = os.environ.get("MODELS_ROOT", "/data/models")',
            'models_root = self._weights_dir if self._weights_dir else os.environ.get("MODELS_ROOT", "/data/models")',
        )
    return text


def build_adapter(model_id: str) -> str:
    """生成单个模型的 adapter.py 内容。"""
    from app.adapters import get_adapter_class

    cls = get_adapter_class(model_id)
    module = sys.modules[cls.__module__]
    source = inspect.getsource(module)
    tree = ast.parse(source)

    cls_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__
    )
    deps = _collect_dependencies(tree, cls.__name__, source)
    body = _transform_class(source, cls_node, model_id)
    body = _special_patches(model_id, cls.__name__, body)

    header = (
        f'"""\n{model_id} 专用适配器（自动生成，请勿手工编辑）。\n'
        "权重加载：优先从本文件夹 weights/ 加载，缺省回退 Hugging Face。\n"
        "输入输出契约与部署平台框架一致（见 server.py 与 weights/README.txt）。\n"
        '"""\n'
    )
    imports = (
        "from typing import Any\n"
        "from server import (\n"
        "    BaseAdapter,\n"
        "    ModelLoadError,\n"
        "    InvalidInputError,\n"
        "    InferenceError,\n"
        "    b64_to_bytes,\n"
        "    bytes_to_b64,\n"
        "    b64_to_image,\n"
        "    image_to_b64,\n"
        "    b64_to_audio,\n"
        "    audio_to_b64,\n"
        "    numpy_audio_to_b64,\n"
        ")\n"
        "\n"
    )
    parts = [header, imports]
    if deps:
        parts.append("\n\n".join(deps))
        parts.append("")
    parts.append(body)
    parts.append(f"\n\n# 统一适配器入口（server.py 加载此名称）\nAdapter = {cls.__name__}\n")
    return "\n".join(parts)


def build_requirements(model_id: str, category: str) -> str:
    """合并 common + 类别依赖 + 模型级附加依赖。"""
    header = f"# 依赖清单：common（平台通用）+ {category}（模型类别）\n"
    common = (REPO_ROOT / "requirements" / "common.txt").read_text(encoding="utf-8")
    cat_file = REPO_ROOT / "requirements" / f"{category}.txt"
    cat_text = cat_file.read_text(encoding="utf-8") if cat_file.exists() else "# （无额外类别依赖）\n"
    parts = [f"{header}\n# ===== common =====\n{common}\n# ===== {category} =====\n{cat_text}"]
    extras = PER_MODEL_REQUIREMENTS.get(model_id)
    if extras:
        parts.append(f"# ===== {model_id} 模型级附加依赖 =====\n" + "\n".join(extras) + "\n")
    return "\n".join(parts)


def generate(output_root: Path) -> None:
    from app.adapters import get_adapter_class, registered_model_ids

    model_ids = sorted(registered_model_ids())
    print(f"[gen] 待生成模型数: {len(model_ids)}")

    server_tpl = (TEMPLATES_DIR / "server.py").read_text(encoding="utf-8")
    start_sh = (TEMPLATES_DIR / "start.sh").read_text(encoding="utf-8")
    start_bat = (TEMPLATES_DIR / "start.bat").read_text(encoding="utf-8")
    weights_readme = (TEMPLATES_DIR / "weights_README.txt").read_text(encoding="utf-8")

    for i, model_id in enumerate(model_ids, 1):
        folder = output_root / model_id.replace("/", "-")
        folder.mkdir(parents=True, exist_ok=True)

        _write_lf(folder / "server.py", server_tpl)
        _write_lf(folder / "start.sh", start_sh)
        _write_lf(folder / "start.bat", start_bat)
        _write_lf(folder / "adapter.py", build_adapter(model_id))

        cls = get_adapter_class(model_id)
        _write_lf(folder / "requirements.txt", build_requirements(model_id, cls.category))

        # 测试脚本：test/test-<模型名>/test-<模型名>.py（验证 /health + /predict）
        test_dir = TEST_OUTPUT / f"test-{folder.name}"
        test_dir.mkdir(parents=True, exist_ok=True)
        _write_lf(test_dir / f"test-{folder.name}.py", build_test_script(model_id, cls.__name__))
        # 测试数据文件预置（脚本运行时只读取，不生成）
        data_dir = test_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for fname, content in build_test_data(cls.__name__).items():
            if fname.endswith((".png", ".wav", ".jpg", ".mp4", ".glb")):
                (data_dir / fname).write_bytes(content)
            else:
                _write_lf(data_dir / fname, content.decode("utf-8"))

        weights_dir = folder / "weights"
        weights_dir.mkdir(exist_ok=True)  # 保留用户已放置的权重
        readme_path = weights_dir / "README.txt"
        if not readme_path.exists():
            _write_lf(readme_path, weights_readme)

        print(f"  [{i}/{len(model_ids)}] {folder.name} (category={cls.category})")

    print(f"[gen] 完成：{len(model_ids)} 个模型文件夹 -> {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 100 个模型隔离部署文件夹")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
