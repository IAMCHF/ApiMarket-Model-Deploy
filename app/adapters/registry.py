# ============================================================================
# 适配器注册表
#
# 注册机制：适配器类通过 @register_adapter() 装饰器登记，
# registry 按 model_id 匹配 MODELS 元组，返回对应的适配器类。
#
# 校验约定：一个 model_id 只能被一个适配器类注册（重复注册视为错误，
# 由 tools 层静态检查发现）。
# ============================================================================

from typing import Any

from .base import BaseAdapter

# 已注册的适配器类（按注册顺序）
_ADAPTER_CLASSES: list[type[BaseAdapter]] = []


def register_adapter(category: str | None = None):
    """类装饰器：登记适配器类。category 可覆盖类属性中的默认值。"""

    def decorator(cls: type[BaseAdapter]) -> type[BaseAdapter]:
        if category is not None:
            cls.category = category
        _ADAPTER_CLASSES.append(cls)
        return cls

    return decorator


def get_adapter_class(model_id: str) -> type[BaseAdapter] | None:
    """按模型 ID 查找适配器类；无匹配返回 None。"""
    for cls in _ADAPTER_CLASSES:
        if not cls.MODELS or model_id in cls.MODELS:
            return cls
    return None


def create_adapter(
    model_id: str,
    device: str = "cuda:0",
    config: dict[str, Any] | None = None,
) -> BaseAdapter:
    """实例化 model_id 对应的适配器；未注册抛 KeyError。"""
    cls = get_adapter_class(model_id)
    if cls is None:
        raise KeyError(f"no adapter registered for model '{model_id}'")
    return cls(model_id=model_id, device=device, config=config)


def registered_model_ids() -> list[str]:
    """所有已注册的模型 ID（用于静态校验 100 个模型全覆盖）。"""
    ids: list[str] = []
    for cls in _ADAPTER_CLASSES:
        ids.extend(cls.MODELS)
    return ids


def adapter_count() -> int:
    """已注册的适配器类数量。"""
    return len(_ADAPTER_CLASSES)


def check_duplicates() -> list[str]:
    """返回被多个适配器重复注册的模型 ID（校验用）。"""
    seen: dict[str, int] = {}
    for model_id in registered_model_ids():
        seen[model_id] = seen.get(model_id, 0) + 1
    return [mid for mid, count in seen.items() if count > 1]
