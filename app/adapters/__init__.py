# ============================================================================
# 适配器包入口：导入全部适配器模块（触发注册），导出工厂函数
#
# 约定：
#   - 所有适配器模块在此处显式导入（保证注册表完整）
#   - 第三方库均为延迟导入（在 _load 内），此处导入零副作用
# ============================================================================

from .base import (
    BaseAdapter,
    InferenceError,
    InvalidInputError,
    ModelLoadError,
)
from .registry import (
    adapter_count,
    check_duplicates,
    create_adapter,
    get_adapter_class,
    registered_model_ids,
)

# 导入全部适配器模块以触发 @register_adapter 注册
from . import (
    asr,          # 语音识别 9
    code_gen,     # 代码生成 3
    detection,    # 分割/检测 8
    embedding,    # 嵌入/重排序 17
    gen3d,        # 3D 生成 6
    image_gen,    # 图像生成 9
    ocr,          # OCR/文档 23
    other,        # 其他 7
    robot,        # 机器人 2
    tts,          # 语音合成 13
    video_gen,    # 视频生成 3
)

__all__ = [
    "BaseAdapter",
    "InferenceError",
    "InvalidInputError",
    "ModelLoadError",
    "adapter_count",
    "check_duplicates",
    "create_adapter",
    "get_adapter_class",
    "registered_model_ids",
]
