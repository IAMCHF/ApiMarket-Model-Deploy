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
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DEFAULT_OUTPUT = REPO_ROOT / "models"

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


def _write_lf(path: Path, text: str) -> None:
    """写文件并强制 LF 行尾（Windows 下避免 CRLF 破坏容器内 bash 脚本）。"""
    path.write_text(text, encoding="utf-8", newline="\n")


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
    # 1. 提取类别（原由 @register_adapter 装饰器设置；源码段不含装饰器）
    category = _extract_category(cls_node)
    text = re.sub(r"^@register_adapter\([^\n]*\)\n", "", text, flags=re.M)
    # 2. MODELS 元组裁剪为当前模型
    text = re.sub(
        r"MODELS\s*=\s*\([^)]*\)",
        f"MODELS = ({model_id!r},)",
        text,
        flags=re.S,
    )
    # 3. 加载引用：from_pretrained(self.model_id / 位置参数 -> self._load_ref
    text = re.sub(
        r"from_pretrained\(\s*self\.model_id",
        "from_pretrained(self._load_ref",
        text,
        flags=re.S,
    )
    # 4. 构造式加载器：SentenceTransformer(self.model_id -> self._load_ref
    text = re.sub(
        r"SentenceTransformer\(\s*self\.model_id",
        "SentenceTransformer(self._load_ref",
        text,
    )
    # 5. 关键字参数加载：model=/repo_id=/model_path=... -> self._load_ref
    kw_alt = "|".join(_LOAD_KWARGS)
    text = re.sub(
        rf"\b({kw_alt})=self\.model_id\b",
        r"\1=self._load_ref",
        text,
    )
    # 6. 内部相对导入改写
    text = _rewrite_io_imports(text)
    # 7. 注入类别（装饰器被剥离后类属性缺失，需显式补回）
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
        # GGUF：优先从本目录 weights/ 扫描 .gguf，缺省回退 MODELS_ROOT
        text = text.replace(
            'models_root = os.environ.get("MODELS_ROOT", "/data/models")\n        model_dir = os.path.join(models_root, self.model_id)',
            'model_dir = self._weights_dir if self._weights_dir else os.path.join(os.environ.get("MODELS_ROOT", "/data/models"), self.model_id)',
        )
        text = text.replace("tools/download_models.py", "weights/ 权重预置")
    elif class_name == "GroundingDINOAdapter":
        # 权重已预置时跳过 snapshot_download，直接从 weights/ 定位 config 与权重
        text = text.replace(
            "repo_dir = snapshot_download(repo_id=self._load_ref)",
            "repo_dir = self._weights_dir if self._weights_dir else snapshot_download(repo_id=self.model_id)",
        )
    elif class_name == "GR00TAdapter":
        # lerobot 权重根目录指向本目录 weights/
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
