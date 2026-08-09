#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# generate_model_folders.py 鈥?涓?100 涓ā鍨嬬敓鎴愬畬鍏ㄩ殧绂荤殑閮ㄧ讲鏂囦欢澶?#
# 杈撳嚭锛歮odels/<缁勭粐鍚?妯″瀷鍚?/
#   server.py        鑷寘鍚湇鍔℃鏋讹紙BaseAdapter + io_codec + FastAPI锛屽浐瀹?0.0.0.0:8080锛?#   adapter.py       璇ユā鍨嬩笓鐢ㄩ€傞厤鍣紙浠?app/adapters 鎻愬彇骞跺崟妯″瀷鍖栵級
#   requirements.txt 渚濊禆娓呭崟锛坈ommon + 绫诲埆锛?#   start.sh         缁熶竴鍚姩鑴氭湰锛圠inux锛?#   start.bat        缁熶竴鍚姩鑴氭湰锛圵indows锛?#   weights/         鏉冮噸鏀剧疆鐩綍锛堢敤鎴烽缃紝鑷姩淇濈暀涓嶈鐩栵級
#
# 鐢ㄦ硶锛歱ython scripts/generate_model_folders.py
# 璇存槑锛氶噸澶嶈繍琛屼細瑕嗙洊浠ｇ爜鏂囦欢锛屼絾涓嶄細鍒犻櫎 weights/ 涓凡鏀剧疆鐨勬潈閲嶃€?# ============================================================================

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

# 鍔犺浇寮曠敤鏇挎崲鐨勫叧閿瓧鍙傛暟鍚嶏紙from_pretrained 绫昏皟鐢ㄧ殑绗竴涓綅缃弬鏁颁害鏇挎崲锛?_LOAD_KWARGS = (
    "model",
    "model_id",
    "model_name",
    "model_path",
    "repo_id",
    "pretrained_path",
    "pretrained_model_name_or_path",
)


def _rewrite_io_imports(text: str) -> str:
    """灏嗙浉瀵瑰鍏ユ敼鍐欎负鎸囧悜鍚岀洰褰?server.py锛堣嚜鍖呭惈闅旂锛夈€?""
    text = re.sub(r"from \.\.utils\.io_codec import", "from server import", text)
    text = re.sub(r"from \.base import", "from server import", text)
    text = re.sub(r"from \.registry import[^\n]*", "", text)
    return text


def _extract_category(cls_node: ast.ClassDef) -> str | None:
    """浠庤楗板櫒 @register_adapter(category=...) 鎻愬彇绫诲埆锛堟簮鐮佹涓嶅惈瑁呴グ鍣紝椤昏蛋 AST锛夈€?""
    for dec in cls_node.decorator_list:
        if (
            isinstance(dec, ast.Call)
            and getattr(dec.func, "id", None) == "register_adapter"
        ):
            for kw in dec.keywords:
                if kw.arg == "category" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def _transform_class(source: str, cls_node: ast.ClassDef, model_id: str) -> str:
    """瀵圭洰鏍囩被婧愮爜鍋氬崟妯″瀷鍖栧彉鎹紙淇濈暀娉ㄩ噴锛夈€?""
    text = ast.get_source_segment(source, cls_node)
    # 1. 鎻愬彇绫诲埆锛堝師鐢?@register_adapter 瑁呴グ鍣ㄨ缃紱婧愮爜娈典笉鍚楗板櫒锛?    category = _extract_category(cls_node)
    text = re.sub(r"^@register_adapter\([^\n]*\)\n", "", text, flags=re.M)
    # 2. MODELS 鍏冪粍瑁佸壀涓哄綋鍓嶆ā鍨?    text = re.sub(
        r"MODELS\s*=\s*\([^)]*\)",
        f"MODELS = ({model_id!r},)",
        text,
        flags=re.S,
    )
    # 3. 鍔犺浇寮曠敤锛歠rom_pretrained(self.model_id / 浣嶇疆鍙傛暟 -> self._load_ref
    text = re.sub(
        r"from_pretrained\(\s*self\.model_id",
        "from_pretrained(self._load_ref",
        text,
        flags=re.S,
    )
    # 4. 鏋勯€犲紡鍔犺浇鍣細SentenceTransformer(self.model_id -> self._load_ref
    text = re.sub(
        r"SentenceTransformer\(\s*self\.model_id",
        "SentenceTransformer(self._load_ref",
        text,
    )
    # 5. 鍏抽敭瀛楀弬鏁板姞杞斤細model=/repo_id=/model_path=... -> self._load_ref
    kw_alt = "|".join(_LOAD_KWARGS)
    text = re.sub(
        rf"\b({kw_alt})=self\.model_id\b",
        r"\1=self._load_ref",
        text,
    )
    # 6. 鍐呴儴鐩稿瀵煎叆鏀瑰啓
    text = _rewrite_io_imports(text)
    # 7. 娉ㄥ叆绫诲埆锛堣楗板櫒琚墺绂诲悗绫诲睘鎬х己澶憋紝闇€鏄惧紡琛ュ洖锛?    if category:
        text = text.rstrip() + f"\n\n    category = {category!r}\n"
    return text


def _collect_dependencies(tree: ast.Module, target: str, source: str) -> list[str]:
    """鏀堕泦鐩爣绫讳緷璧栫殑妯″潡绾у嚱鏁?绫伙紙鍩虹被 mixin銆乭elper 鍑芥暟锛夛紝鎸夋簮鐮侀『搴忚繑鍥炪€?""
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
    """瀵规棤娉曢€氱敤鏇挎崲鐨勫姞杞介€昏緫鍋氬畾鍚戣ˉ涓侊紙鎸囧悜鏈湴 weights/锛夈€?""
    if class_name == "LlamaCppCodeGenAdapter":
        # GGUF锛氫紭鍏堜粠鏈洰褰?weights/ 鎵弿 .gguf锛岀己鐪佸洖閫€ MODELS_ROOT
        text = text.replace(
            'models_root = os.environ.get("MODELS_ROOT", "/data/models")\n        model_dir = os.path.join(models_root, self.model_id)',
            'model_dir = self._weights_dir if self._weights_dir else os.path.join(os.environ.get("MODELS_ROOT", "/data/models"), self.model_id)',
        )
        text = text.replace("tools/download_models.py", "weights/ 鏉冮噸棰勭疆")
    elif class_name == "GroundingDINOAdapter":
        # 鏉冮噸宸查缃椂璺宠繃 snapshot_download锛岀洿鎺ヤ粠 weights/ 瀹氫綅 config 涓庢潈閲?        text = text.replace(
            "repo_dir = snapshot_download(repo_id=self._load_ref)",
            "repo_dir = self._weights_dir if self._weights_dir else snapshot_download(repo_id=self.model_id)",
        )
    elif class_name == "GR00TAdapter":
        # lerobot 鏉冮噸鏍圭洰褰曟寚鍚戞湰鐩綍 weights/
        text = text.replace(
            'models_root = os.environ.get("MODELS_ROOT", "/data/models")',
            'models_root = self._weights_dir if self._weights_dir else os.environ.get("MODELS_ROOT", "/data/models")',
        )
    return text


def build_adapter(model_id: str) -> str:
    """鐢熸垚鍗曚釜妯″瀷鐨?adapter.py 鍐呭銆?""
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
        f'"""\n{model_id} 涓撶敤閫傞厤鍣紙鑷姩鐢熸垚锛岃鍕挎墜宸ョ紪杈戯級銆俓n'
        "鏉冮噸鍔犺浇锛氫紭鍏堜粠鏈枃浠跺す weights/ 鍔犺浇锛岀己鐪佸洖閫€ Hugging Face銆俓n"
        "杈撳叆杈撳嚭濂戠害涓庨儴缃插钩鍙版鏋朵竴鑷达紙瑙?server.py 涓?weights/README.txt锛夈€俓n"
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
    parts.append(f"\n\n# 缁熶竴閫傞厤鍣ㄥ叆鍙ｏ紙server.py 鍔犺浇姝ゅ悕绉帮級\nAdapter = {cls.__name__}\n")
    return "\n".join(parts)


def build_requirements(category: str) -> str:
    """鍚堝苟 common + 绫诲埆渚濊禆銆?""
    header = f"# 渚濊禆娓呭崟锛歝ommon锛堝钩鍙伴€氱敤锛? {category}锛堟ā鍨嬬被鍒級\n"
    common = (REPO_ROOT / "requirements" / "common.txt").read_text(encoding="utf-8")
    cat_file = REPO_ROOT / "requirements" / f"{category}.txt"
    cat_text = cat_file.read_text(encoding="utf-8") if cat_file.exists() else "# 锛堟棤棰濆绫诲埆渚濊禆锛塡n"
    return f"{header}\n# ===== common =====\n{common}\n# ===== {category} =====\n{cat_text}"


def generate(output_root: Path) -> None:
    from app.adapters import get_adapter_class, registered_model_ids

    model_ids = sorted(registered_model_ids())
    print(f"[gen] 寰呯敓鎴愭ā鍨嬫暟: {len(model_ids)}")

    server_tpl = (TEMPLATES_DIR / "server.py").read_text(encoding="utf-8")
    start_sh = (TEMPLATES_DIR / "start.sh").read_text(encoding="utf-8")
    start_bat = (TEMPLATES_DIR / "start.bat").read_text(encoding="utf-8")
    weights_readme = (TEMPLATES_DIR / "weights_README.txt").read_text(encoding="utf-8")

    for i, model_id in enumerate(model_ids, 1):
        folder = output_root / model_id.replace("/", "-")
        folder.mkdir(parents=True, exist_ok=True)

        (folder / "server.py").write_text(server_tpl, encoding="utf-8", newline="\n")
        (folder / "start.sh").write_text(start_sh, encoding="utf-8", newline="\n")
        (folder / "start.bat").write_text(start_bat, encoding="utf-8", newline="\n")
        (folder / "adapter.py").write_text(build_adapter(model_id), encoding="utf-8", newline="\n")

        cls = get_adapter_class(model_id)
        (folder / "requirements.txt").write_text(build_requirements(cls.category), encoding="utf-8", newline="\n")

        weights_dir = folder / "weights"
        weights_dir.mkdir(exist_ok=True)  # 淇濈暀鐢ㄦ埛宸叉斁缃殑鏉冮噸
        readme_path = weights_dir / "README.txt"
        if not readme_path.exists():
            readme_path.write_text(weights_readme, encoding="utf-8", newline="\n")

        print(f"  [{i}/{len(model_ids)}] {folder.name} (category={cls.category})")

    print(f"[gen] 瀹屾垚锛歿len(model_ids)} 涓ā鍨嬫枃浠跺す -> {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="鐢熸垚 100 涓ā鍨嬮殧绂婚儴缃叉枃浠跺す")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
