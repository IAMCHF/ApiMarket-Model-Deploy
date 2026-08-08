# ============================================================================
# 具身智能（Embodied AI）适配器（2 个模型）
#
# 覆盖 NVIDIA GR00T 系列的机器人控制策略：
#   1. nvidia/GR00T-N1.7-3B     —— 人形机器人全身控制（NVIDIA GR00T，lerobot 生态）
#   2. nvidia/GR00T-N1.7-DROID  —— VLA 灵巧操作策略（DeX 级操作，lerobot 生态）
#
# 运行时说明：
#   NVIDIA GR00T N1.7 通过 lerobot 的 gr00t 环境提供服务（策略权重与输入输出
#   格式均遵循 lerobot 约定）。真实机器人部署需要 NVIDIA Isaac Lab / 实体
#   机器人运行时（如 Go1 / GR00T 系列本体）配合 —— 当前 /predict 返回的是
#   占位动作向量，仅用于打通平台推理链路。
#
# /predict 输入输出约定：
#   输入：{"image": "<base64>", "instruction": "抓取红色杯子"}
#   输出：{"action": [0.0, ...], "action_dim": 64}
# ============================================================================

import os
from typing import Any

from .base import BaseAdapter, InvalidInputError, ModelLoadError
from .registry import register_adapter
from ..utils.io_codec import b64_to_image


@register_adapter(category="robot")
class GR00TAdapter(BaseAdapter):
    """NVIDIA GR00T N1.7 机器人控制策略适配器。

    MODELS 覆盖人形全身控制（-3B）与 DROID VLA 灵巧操作两类策略，
    二者共用 lerobot 的 gr00t 加载路径。
    """

    MODELS = (
        "nvidia/GR00T-N1.7-3B",
        "nvidia/GR00T-N1.7-DROID",
    )

    # 各模型的默认动作维度（占位用；上线前按策略配置校准）
    _DEFAULT_ACTION_DIM = {
        "nvidia/GR00T-N1.7-3B": 64,       # 全身控制：关节角度 + 基座速度等
        "nvidia/GR00T-N1.7-DROID": 7,     # 单臂 VLA 操作：末端位姿 + 夹爪
    }

    def _load(self) -> None:
        """延迟导入 lerobot 栈并加载 GR00T 策略（best-effort）。

        TODO(上线前必做)：
          1. 确认部署镜像中的 lerobot 版本，校准 make_policy 的调用签名
             （NVIDIA GR00T N1.7 通过 lerobot 的 gr00t 策略实现提供服务，
              policy name 应为 "gr00t"）。
          2. 权重加载方式二选一：
             a) 本地权重：从 self._models_root/<HF_REPO_ID> 读取；
             b) HF Hub：由 tools/download_models.py 预先拉取到 /data/models，
                再经 hf_hub_download 定位。
          3. 校验策略 cfg 中 GR00T-N1.7 系列需要的字段（pretrained_path、
             action chunking 等），并据此初始化自回归动作生成循环。
          4. 若需要，叠加 NVIDIA Isaac Lab 运行时做 sim2real 推理。
        """
        models_root = os.environ.get("MODELS_ROOT", "/data/models")

        try:
            import torch  # noqa: F401 - 与 lerobot 共享 torch 依赖
            # lerobot 策略工厂：根据 policy name 构建对应的策略实例
            from lerobot.common.policies.factory import make_policy
        except ImportError as exc:
            raise ModelLoadError(
                f"依赖缺失：lerobot / torch 未安装，无法加载 GR00T 策略（{exc}）"
            ) from exc

        # best-effort 加载路径：先尝试按 lerobot 官方 gr00t 约定加载，
        # 若签名或权重布局与当前 lerobot 版本不一致，则抛 NotImplementedError
        # 由后续 TODO 中的精确实现接管。
        try:
            pretrained_path = os.path.join(models_root, self.model_id)
            self._policy = make_policy(
                name="gr00t",
                pretrained_path=pretrained_path,
            )
            self._policy.eval()
        except Exception as exc:  # noqa: BLE001 - 签名未校准，转为骨架错误
            raise NotImplementedError(
                f"需接入 NVIDIA GR00T 运行时 — see TODO（当前 lerobot 加载路径未校准：{exc}）"
            ) from exc

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """输入图像 + 指令 -> 占位动作向量。

        当前仅返回定长占位动作向量，用于验证推理链路；
        真实机器人部署需接入 NVIDIA Isaac / lerobot 运行时后替换。
        """
        image_b64 = inputs.get("image")
        instruction = inputs.get("instruction")
        if not isinstance(image_b64, str) or not image_b64:
            raise InvalidInputError("inputs 必须包含非空的 'image'（base64 字符串）")
        if not isinstance(instruction, str) or not instruction.strip():
            raise InvalidInputError("inputs 必须包含非空的 'instruction'（字符串）")

        # 校验图像可解码（格式/损坏检查），失败即返回 400
        try:
            b64_to_image(image_b64)
        except Exception as exc:  # noqa: BLE001 - 交给上层转 400 INVALID_INPUT
            raise InvalidInputError(f"'image' 不是合法的 base64 图像：{exc}") from exc

        action_dim = int(
            params.get(
                "action_dim",
                self.config.get(
                    "action_dim", self._DEFAULT_ACTION_DIM.get(self.model_id, 64)
                ),
            )
        )
        if action_dim <= 0:
            raise InvalidInputError("'action_dim' 必须为正整数")

        # TODO(真实部署)：以下为占位动作 —— 应将图像 + 指令送入已加载的
        # gr00t 策略（self._policy），经 tokenization / 自回归解码得到真实动作。
        # 示例如下（按 lerobot 版本校准）：
        #   with torch.inference_mode():
        #       obs = _encode_observation(image, instruction)   # TODO
        #       action = self._policy.select_action(obs)        # TODO
        placeholder_action = [0.0] * action_dim

        return {
            "action": placeholder_action,
            "action_dim": action_dim,
        }
