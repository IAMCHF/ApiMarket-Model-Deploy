# ============================================================================
# GPU 信息查询（nvidia-smi 子进程，避免引入 pynvml 依赖）
# ============================================================================

import shutil
import subprocess
from typing import Any


def _nvidia_smi() -> list[dict[str, Any]] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001 - 查询失败视为无 GPU 信息
        return None

    infos: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        infos.append(
            {
                "id": int(parts[0]),
                "name": parts[1],
                "vram_total_gb": round(int(parts[2]) / 1024, 1),
                "vram_used_gb": round(int(parts[3]) / 1024, 1),
                "gpu_util_pct": int(parts[4]),
            }
        )
    return infos


def get_gpu_info() -> dict[str, Any]:
    """返回 /health 使用的 GPU 汇总信息。"""
    infos = _nvidia_smi()
    return {"available": infos is not None, "devices": infos or []}
