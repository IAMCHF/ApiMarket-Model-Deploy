# ============================================================================
# 校验脚本（开发期使用）：验证 100 个模型全部注册、无重复、与 models.yaml 对齐
# 用法：python scripts/verify_registry.py
# ============================================================================

import os
import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，使 app 成为可导入包
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    import yaml

    from app.adapters import adapter_count, check_duplicates, create_adapter, registered_model_ids

    # 1. 适配器注册表检查
    registered = set(registered_model_ids())
    dups = check_duplicates()
    print(f"[1] 适配器类数量: {adapter_count()}")
    print(f"[1] 注册模型数量: {len(registered)}")
    if dups:
        print(f"[1] !! 重复注册: {dups}")
        return 1

    # 2. models.yaml 模型清单
    config_path = REPO_ROOT / "config" / "models.yaml"
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    yaml_models = set(data.get("models", {}).keys())
    print(f"[2] models.yaml 模型数量: {len(yaml_models)}")

    # 3. 双向对齐检查
    missing_in_registry = yaml_models - registered
    missing_in_yaml = registered - yaml_models
    if missing_in_registry:
        print(f"[3] !! 已配置但未注册适配器: {sorted(missing_in_registry)}")
        return 1
    if missing_in_yaml:
        print(f"[3] !! 已注册但未写入 models.yaml: {sorted(missing_in_yaml)}")
        return 1
    print(f"[3] 注册表与 models.yaml 完全对齐（{len(registered)} 个模型）")

    # 4. 每个模型可实例化
    for model_id in sorted(registered):
        try:
            create_adapter(model_id, device="cpu", config={})
        except Exception as exc:  # noqa: BLE001
            print(f"[4] !! 实例化失败: {model_id}: {exc}")
            return 1
    print("[4] 全部模型可实例化")

    # 5. 类别分布统计
    from app.adapters import get_adapter_class

    counts: dict[str, int] = {}
    for model_id in sorted(registered):
        cls = get_adapter_class(model_id)
        counts[cls.category] = counts.get(cls.category, 0) + 1
    print(f"[5] 类别分布: {dict(sorted(counts.items()))}")
    print("\n== 校验通过 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
