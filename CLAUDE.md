# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

单镜像通用 AI 模型部署平台：100 个精选模型（OCR/ASR/TTS/图像/视频/3D/嵌入/检测/机器人/代码/其他，共 11 类）共用同一 Docker 基础镜像，通过统一 HTTP 信封（`/health` + `/predict`）对外服务。

方案基线文档为根目录 `AI模型部署平台方案.md`（镜像分层、CI/CD、约束），模型筛选历程见 `精选模型.md` / `参考.md`。仓库语言为中文，注释与文档一律使用中文。

## 核心架构：镜像 vs 运行时挂载

**镜像（变更频率极低）**：`Dockerfile` + `requirements-base.txt` 固化了 CUDA 12.1 + Python 3.11 + PyTorch 2.4 (cu121) + transformers/accelerate 等通用框架。镜像内**没有** requirements、脚本、业务代码、模型权重，也无 ENTRYPOINT/CMD，启动命令完全由外部注入。镜像重建只发生在 Dockerfile / requirements-base.txt 变更时（由 `.github/workflows/build-image.yml` 构建并导出为 tar.gz artifact 供内网加载）。

**运行时挂载（高频修改，改后重启容器即生效，无需重建镜像）**：
- `config/models.yaml` — 100 个 model_id → `{category, vram_gb}` 映射
- `requirements/{category}.txt` — 按类别的专用 pip 依赖（另有 `common.txt` 所有服务必装）
- `scripts/` — entrypoint.sh（装包→下载权重→拉起 uvicorn）、download_models.py、healthcheck.py
- `app/` — FastAPI 业务框架

容器运行时通过环境变量区分服务：`MODEL_TYPE`（选 requirements 文件）、`MODEL_ID`（选权重与适配器）、`SERVICE_PORT`、`DEVICE`、`HF_ENDPOINT`（国内镜像）、`MODELS_ROOT`（权重持久卷，默认 `/data/models`）、`MODELS_CONFIG`。挂载路径约定见 `docker-compose.yml` 与方案文档 6.3。

## 适配器框架（app/adapters/）

每个模型由一个适配器类实现，通过 `@register_adapter(category=...)` 装饰器注册，`registry.py` 按 `MODELS` 元组匹配 model_id。**硬性约定**：

1. **所有第三方库（torch/transformers/diffusers/paddle 等）必须在 `_load()` 内延迟导入**——保证框架代码可在未装专用依赖的环境被导入和静态校验（`app/adapters/__init__.py` 顶层导入全部适配器模块触发注册）。
2. 子类只需实现 `_load()` 与 `_predict()`；`BaseAdapter.predict()` 是线程安全模板方法（自动加载 + 串行化推理），`load()` 是幂等的。
3. `category` 类属性必须与 `requirements/{category}.txt` 文件名一一对应。
4. 所有注册的 `MODELS` 必须与 `config/models.yaml` 完全对齐（`verify_registry.py` 强制检查，双向）。

## HTTP 契约

- `GET /health`：`loading`(503) / `ready`(200) / `error`(500)，含 GPU 信息与适配器状态
- `POST /predict`：请求 `{"inputs": {...}, "params": {...}}`，响应 `{"outputs": ..., "latency_ms": ..., "model": ...}`；文件类输入输出一律 base64（`app/utils/io_codec.py` 编解码）
- 错误信封：`{"detail": {"code": ..., "message": ...}}`，code 为 `INVALID_INPUT`(400) / `MODEL_LOAD_FAILED`(503) / `INFERENCE_ERROR`(500) / `INTERNAL_ERROR`(500)，异常类型定义在 `app/adapters/base.py`

## 双部署形态

1. **容器平台模式**：entrypoint.sh → uvicorn 加载 `app/server.py`（平台内所有模型共用一个框架进程，按 `MODEL_ID` 选适配器）。
2. **单模型隔离模式**：`scripts/generate_model_folders.py` 从适配器注册表 + `scripts/templates/` 为每个模型生成 `models/<组织-模型名>/` 自包含文件夹（server.py 模板 + 单模型化 adapter.py + requirements.txt + start.sh/start.bat + weights/）。**生成产物勿手工编辑**——重复运行会覆盖代码文件，但不会删除 `weights/` 中用户已放置的权重。生成器对 GGUF、GroundingDINO、GR00T 等有特殊补丁逻辑（`_special_patches`），修改生成逻辑时需同步验证。

## 常用命令

```bash
# 校验：适配器注册 ↔ models.yaml 双向对齐、无重复注册、全部可实例化（仓库唯一的"测试"）
python scripts/verify_registry.py

# 重新生成 100 个单模型部署文件夹（models/）
python scripts/generate_model_folders.py

# 本地开发起服务（需先设置 MODEL_ID / MODEL_TYPE；未注册的 MODEL_ID 服务也能起，/health /predict 返回明确错误）
MODEL_ID=baidu/Unlimited-OCR MODEL_TYPE=ocr python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

# 构建基础镜像
docker build -t model-deploy:1.0 .

# 容器化部署（镜像 + 挂载 config/requirements/scripts/app）
docker compose up -d --force-recreate   # 见 docker-compose.yml，使用 image: model-deploy:1.0
```

无测试框架、无 lint 配置；改动后以 `verify_registry.py` + 容器内 `/health`（`scripts/healthcheck.py` 即探针）验证。

## 注意事项

- 新增模型 = 三步：适配器类 + `config/models.yaml` 条目 + （必要时）新 requirements 文件；完成后跑 `verify_registry.py`。类别 key 变更会破坏 entrypoint 的装包映射。
- `models/` 下 100 个文件夹全部由生成器产出，与其在 `app/adapters/` 的源保持一致；单模型改 bug 应改源头适配器再重新生成，而不是直接改 `models/` 产物。
- 显存估算：FP16 ≈ 参数量(B)×2 GB，INT8 ≈ 50%，INT4 ≈ 25%（方案文档 8.3）；单服务 ≤ 40GB 是硬约束。
- 国内网络部署设置 `HF_ENDPOINT=https://hf-mirror.com`。
