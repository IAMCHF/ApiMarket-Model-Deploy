# ============================================================================
# 3D 生成 / 重建适配器（6 个模型）
#
# 六个适配器族：
#   1. TRELLISAdapter      —— microsoft/TRELLIS.2-4B 图像转 3D（trellis 管线）
#   2. Hunyuan3DAdapter    —— tencent/Hunyuan3D-2.1 图像转 3D（hunyuan3d 官方模块）
#   3. StableFast3DAdapter —— stabilityai/stable-fast-3d 图像转 UV 网格（sf3d）
#   4. TripoSGAdapter      —— VAST-AI/TripoSG 图像转网格（triposg 官方）
#   5. InstantNuRecAdapter —— nvidia/instant-nurec 神经场景重建（骨架，待接入）
#   6. Step1X3DAdapter     —— stepfun-ai/Step1X-3D 图像转 3D（骨架，待接入）
#
# /predict 输入输出约定：
#   输入：{"image": "<base64>"}
#   输出：{"model": "<base64 glb>", "format": "glb"}
# ============================================================================

from typing import Any

from .base import BaseAdapter, InvalidInputError
from .registry import register_adapter
from ..utils.io_codec import b64_to_image, bytes_to_b64


def _mesh_to_glb(mesh: Any) -> str:
    """网格对象（trimesh 或带 export 的官方结果对象）-> base64 GLB 字符串。

    优先调用对象自带的 export()；缺失时退化为用 trimesh.Trimesh 重建导出。
    """
    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    try:
        exporter = getattr(mesh, "export", None)
        if exporter is None:
            import trimesh

            trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces).export(tmp_path)
        else:
            mesh.export(tmp_path)  # 扩展名 .glb 决定导出格式（内部经 trimesh/assimp）
        with open(tmp_path, "rb") as fh:
            return bytes_to_b64(fh.read())
    finally:
        os.unlink(tmp_path)


@register_adapter(category="3d-gen")
class TRELLISAdapter(BaseAdapter):
    """microsoft/TRELLIS.2-4B 图像转 3D（trellis 官方管线）。

    使用 trellis.pipelines.TrellisImageTo3DPipeline 推理，
    提取 outputs["mesh"] 导出为 GLB（内部经 trimesh）。

    TODO：TRELLIS.2 管线名 / from_pretrained 入口 / run 返回值
    结构按官方 README 校准后上线。
    """

    MODELS = ("microsoft/TRELLIS.2-4B",)

    def _load(self) -> None:
        # 官方管线：https://github.com/microsoft/TRELLIS
        from trellis.pipelines import TrellisImageTo3DPipeline

        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_id)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        outputs = self._pipeline.run(image, seed=int(params.get("seed", 0)))
        mesh = outputs["mesh"]
        return {"model": _mesh_to_glb(mesh), "format": "glb"}


@register_adapter(category="3d-gen")
class Hunyuan3DAdapter(BaseAdapter):
    """tencent/Hunyuan3D-2.1 图像转 3D（hunyuan3d 官方模块）。

    官方推理入口为 hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline；
    Hunyuan3D-2.1 的管线类名、输入参数与输出对象按官方 README 校准。

    TODO：Hunyuan3D-2.1 具体 API 与网格导出方式待校准。
    """

    MODELS = ("tencent/Hunyuan3D-2.1",)

    def _load(self) -> None:
        # 官方推理：https://github.com/Tencent/Hunyuan3D-2
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_id)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        # TODO: 管线输出对象（dict / 网格 / 元组）按官方 README 校准
        outputs = self._pipeline(image=image)
        mesh = outputs.get("mesh", outputs) if isinstance(outputs, dict) else outputs
        return {"model": _mesh_to_glb(mesh), "format": "glb"}


@register_adapter(category="3d-gen")
class StableFast3DAdapter(BaseAdapter):
    """stabilityai/stable-fast-3d 图像转 UV 网格（sf3d 官方 API）。

    官方推理：from sf3d import SF3D；SF3D.from_pretrained(model_id) 加载，
    generate_mesh(image) 返回（trimesh 网格, 材质）元组或网格对象。
    """

    MODELS = ("stabilityai/stable-fast-3d",)

    def _load(self) -> None:
        # 官方推理：https://github.com/Stability-AI/stable-fast-3d
        from sf3d import SF3D

        self._model = SF3D.from_pretrained(self.model_id, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        mesh = self._model.generate_mesh(image)
        if isinstance(mesh, (tuple, list)):
            mesh = mesh[0]  # 官方返回 (mesh, material)，取网格部分
        return {"model": _mesh_to_glb(mesh), "format": "glb"}


@register_adapter(category="3d-gen")
class TripoSGAdapter(BaseAdapter):
    """VAST-AI/TripoSG 图像转网格（triposg 官方 API）。

    官方推理代码位于 VAST-AI-Research/TripoSG 仓库（src/triposg），
    需将仓库根目录加入 PYTHONPATH 后导入 TripoSG 管线。

    TODO：TripoSG 管线类名 / 加载方式 / generate_mesh 返回值按官方 README 校准。
    """

    MODELS = ("VAST-AI/TripoSG",)

    def _load(self) -> None:
        # 官方推理：https://github.com/VAST-AI-Research/TripoSG
        # TODO: 官方代码位于仓库 src/triposg 下，导入路径按部署方式校准
        from triposg import TripoSG

        self._model = TripoSG.from_pretrained(self.model_id, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        mesh = self._model.generate_mesh(image)
        if isinstance(mesh, (tuple, list)):
            mesh = mesh[0]
        return {"model": _mesh_to_glb(mesh), "format": "glb"}


@register_adapter(category="3d-gen")
class InstantNuRecAdapter(BaseAdapter):
    """nvidia/instant-nurec 神经场景重建。

    官方推理代码待接入 — see TODO。
    待接入内容：instant-nurec 官方重建管线（多视角图像输入、
    体素/网格导出流程按官方 README 校准）。
    """

    MODELS = ("nvidia/instant-nurec",)

    def _load(self) -> None:
        raise NotImplementedError("nvidia/instant-nurec 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("nvidia/instant-nurec 官方推理代码待接入 — see TODO")


@register_adapter(category="3d-gen")
class Step1X3DAdapter(BaseAdapter):
    """stepfun-ai/Step1X-3D 图像转 3D。

    官方推理代码待接入 — see TODO。
    待接入内容：Step1X-3D 官方推理脚本（单视角图像 -> 3D 网格，
    导出管线按官方 README 校准）。
    """

    MODELS = ("stepfun-ai/Step1X-3D",)

    def _load(self) -> None:
        raise NotImplementedError("stepfun-ai/Step1X-3D 官方推理代码待接入 — see TODO")

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("stepfun-ai/Step1X-3D 官方推理代码待接入 — see TODO")
