"""
cisco-ai/Antares-1B 专用适配器（自动生成，请勿手工编辑）。
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


class AntaresAdapter(BaseAdapter):
    """Cisco Antares-1B 代码漏洞定位 / 安全检测。

    注意：Antares 专为气隙（air-gapped）安全审计场景设计，部署环境通常
    无法联网拉取权重，须预先在隔离网络内把模型权重镜像到 $MODELS_ROOT。
    当前为骨架实现 —— 真实推理需接入 Cisco Antares 官方推理运行时。
    """

    MODELS = ('cisco-ai/Antares-1B',)

    def _load(self) -> None:
        """延迟导入 Cisco Antares 官方运行时并加载模型。

        TODO(上线前必做)：
          1. 确认 Antares 官方推理运行时（pip 包 / Docker 镜像）的导入路径，
             在气隙环境中以镜像方式离线安装；
          2. 权重由气隙环境内预置到 $MODELS_ROOT/cisco-ai/Antares-1B；
          3. 校准漏洞定位的输入格式（源码片段 / 文件路径）与输出结构
             （漏洞位置、严重度、CWE 类别等）。
        """
        raise NotImplementedError("需接入 Cisco Antares 运行时 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        # 骨架占位：仅做输入校验，真实推理见 _load 中的 TODO
        _ = inputs, params
        raise NotImplementedError("需接入 Cisco Antares 运行时 — see TODO")

    category = 'other'



# 统一适配器入口（server.py 加载此名称）
Adapter = AntaresAdapter
