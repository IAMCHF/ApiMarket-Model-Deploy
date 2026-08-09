#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# check_requirements.py — 依赖完备性核对（方案A）
#
# 方法：扫描 app/adapters/*.py 中 _load/_predict 内 import 的第三方包，
#       对照 镜像L2固化依赖 + requirements/{category}.txt，
#       输出各类别缺失的依赖清单。
#
# 用法：python scripts/check_requirements.py
# ============================================================================

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 镜像 L2 固化的顶层包（Dockerfile + requirements-base.txt）
L2_PACKAGES = {
    "torch", "torchvision", "torchaudio", "transformers", "accelerate",
    "huggingface_hub", "safetensors", "bitsandbytes", "onnxruntime",
    "numpy", "scipy", "pillow", "einops", "sentencepiece", "protobuf",
    "pyyaml", "requests", "tqdm", "PIL",
}

# 常见标准库（忽略）
STDLIB = {
    "os", "sys", "io", "json", "re", "time", "math", "glob", "shutil",
    "subprocess", "tempfile", "base64", "argparse", "importlib",
    "threading", "abc", "typing", "pathlib", "urllib", "contextlib",
    "functools", "collections", "copy", "itertools", "warnings", "logging",
    "wave", "struct", "random", "dataclasses", "enum", "string", "textwrap",
    "socket", "http", "ssl", "hashlib", "gzip", "zipfile", "tarfile",
}


def extract_imports(source: str) -> set[str]:
    """从源码提取所有 import 的顶层包名。"""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue  # 相对导入（from .base / from ..utils 等）非第三方包
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def category_of(model_id: str) -> str:
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT))
    from app.adapters import get_adapter_class

    cls = get_adapter_class(model_id)
    return cls.category


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from app.adapters import get_adapter_class, registered_model_ids

    # 模型 -> 类别 与 类别 -> 需要的包
    adapter_dir = REPO_ROOT / "app" / "adapters"
    category_packages: dict[str, set[str]] = {}
    model_category: dict[str, str] = {}
    for model_id in registered_model_ids():
        cls = get_adapter_class(model_id)
        model_category[model_id] = cls.category
        # 该模型适配器类的源码（含依赖的模块级 helper）
        mod = sys.modules[cls.__module__]
        import inspect

        try:
            src = inspect.getsource(mod)
        except Exception:
            continue
        pkgs = extract_imports(src) - L2_PACKAGES - STDLIB
        category_packages.setdefault(cls.category, set()).update(pkgs)

    print("=== 各类别适配器 import 的第三方包（对照 requirements） ===\n")
    missing_total = 0
    for category in sorted(category_packages):
        req_file = REPO_ROOT / "requirements" / f"{category}.txt"
        req_text = req_file.read_text(encoding="utf-8") if req_file.exists() else ""
        # requirements 中已出现的包名
        req_pkgs = set()
        for line in req_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_\-\[\]]+)", line)
            if m:
                req_pkgs.add(m.group(1).split("[")[0].lower())
        imported = category_packages[category]
        # 别名映射：import 名 -> PyPI 包名（含常见大小写/连字符差异）
        alias_map = {
            "cv2": "opencv", "PIL": "pillow", "fitz": "pymupdf",
            "llama_cpp": "llama-cpp-python", "sklearn": "scikit-learn",
            "nemo": "nemo_toolkit", "chronos": "chronos-forecasting",
            "imageio": "imageio", "pandas": "pandas",
        }
        missing = []
        for pkg in sorted(imported):
            pkg_name = alias_map.get(pkg, pkg.replace("_", "-")).lower()
            if pkg_name in req_pkgs or pkg.lower() in req_pkgs:
                continue
            missing.append(f"{pkg}（import 名）")
        print(f"[{category}] 适配器 import: {sorted(imported)}")
        if missing:
            print(f"   [缺失可能] {missing}")
            missing_total += 1
        else:
            print("   [无缺失]")
        print()
    print(f"结论：{missing_total} 个类别存在可能缺失的依赖（需人工核对包名与版本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
