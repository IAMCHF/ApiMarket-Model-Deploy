"""
stabilityai/stable-fast-3d 专用适配器（自动生成，请勿手工编辑）。
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

class StableFast3DAdapter(BaseAdapter):
    """stabilityai/stable-fast-3d 图像转 UV 网格（sf3d 官方 API）。

    官方推理：from sf3d import SF3D；SF3D.from_pretrained(model_id) 加载，
    generate_mesh(image) 返回（trimesh 网格, 材质）元组或网格对象。
    """

    MODELS = ('stabilityai/stable-fast-3d',)

    def _load(self) -> None:
        # 官方推理：https://github.com/Stability-AI/stable-fast-3d
        from sf3d import SF3D

        self._model = SF3D.from_pretrained(self._load_ref, device=self.device)

    def _predict(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image_b64 = inputs.get("image")
        if not image_b64:
            raise InvalidInputError("inputs 必须包含 'image'（base64 编码的图像）")
        image = b64_to_image(image_b64)

        mesh = self._model.generate_mesh(image)
        if isinstance(mesh, (tuple, list)):
            mesh = mesh[0]  # 官方返回 (mesh, material)，取网格部分
        return {"model": _mesh_to_glb(mesh), "format": "glb"}

    category = '3d-gen'



# 统一适配器入口（server.py 加载此名称）
Adapter = StableFast3DAdapter
