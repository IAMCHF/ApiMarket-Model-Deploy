# AI 模型部署平台方案文档

> **用途**：本文档作为新建项目的方案基线，用于镜像构建与 CI/CD 持续集成的依据，支持持续修改迭代。
>
> **最后更新**：2026-08-08
>
> **状态**：方案确认中

---

## 目录

1. [项目概述](#1-项目概述)
2. [云服务器资源约束](#2-云服务器资源约束)
3. [内网部署实用性分析与筛选（120 → 100）](#3-内网部署实用性分析与筛选120--100)
4. [精选模型清单（100个）](#4-精选模型清单100个)
5. [分类统计](#5-分类统计)
6. [基础镜像方案](#6-基础镜像方案)
7. [CI/CD 持续集成方案](#7-cicd-持续集成方案)
8. [注意事项与已知约束](#8-注意事项与已知约束)

---

## 1. 项目概述

### 1.1 目标

基于 Hugging Face Trending 榜单筛选及多轮补充得到的 **120 个候选模型**，经**内网部署实用性分析**筛选出 **100 个推荐部署的专用 AI 模型**，构建**单镜像通用部署平台**，实现：

- 一个镜像覆盖所有模型的部署需求
- 模型权重与业务配置与镜像解耦，放持久化存储
- 专用 pip 依赖在容器启动时按需安装
- 镜像构建一次后稳定复用，支持 CI/CD 持续迭代

### 1.2 模型来源与筛选历程

| 迭代轮次 | 操作 | 数量变化 |
|----------|------|----------|
| 初始筛选 | HF Trending 100 个模型 | 100 |
| 第一轮筛选 | 移除 44 个"建议观望"模型，保留 56 个 | 56 |
| 第一轮补充 | 搜集 53 个生产级模型 | 109 |
| 第二轮筛选 | 移除 3 个不符合企业内网标准的模型 | 106 |
| 第二轮补充 | 新增 14 个企业级模型 | **120** |
| **内网实用性再筛选** | **移除 20 个实用性不足模型（本文档第 3 章）** | **100** |

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| 镜像最小化 | 镜像只固化二进制运行环境（CUDA + Python + PyTorch + 通用框架），不含任何配置文件、脚本、requirements |
| 权重外挂 | 模型权重放持久化卷，按需加载，不入镜像 |
| 依赖外挂 | 按模型类别的 requirements 文件放宿主机，挂载进容器，运行时安装 |
| 零配置镜像 | 镜像内无 ENTRYPOINT/CMD/业务文件，启动命令完全由外部注入 |
| 单镜像复用 | 所有模型服务共用同一镜像，差异通过环境变量和挂载点区分 |

### 1.4 GPU 驱动说明

GPU 驱动**不在镜像内**，安装在**宿主机**上，通过 NVIDIA Container Toolkit 在容器启动时注入。镜像内仅包含 CUDA 运行时库（基于 `nvidia/cuda` 官方镜像）。

---

## 2. 云服务器资源约束

### 2.1 硬件约束

| 资源 | 约束 | 说明 |
|------|------|------|
| **GPU 显存** | 单服务 ≤ 40GB | 核心约束。所有模型推理显存需求 ≤ 40GB（FP16 估算，量化版更省） |
| CPU 核数 | 无硬性限制 | 无特殊要求 |
| 内存 | 无硬性限制 | 无特殊要求 |
| 磁盘空间 | 无硬性限制 | 模型权重和 pip 缓存走持久化卷，按需扩展 |

### 2.2 显存约束对模型筛选的影响

- 所有 100 个推荐模型的显存需求均在 40GB 以内，**无 40GB 触顶模型**
- 最大显存需求模型：`deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`（~32GB FP16）、`Qwen/Qwen3-Coder-30B-A3B-Instruct`（~31GB FP16）、`stepfun-ai/Step1X-Edit`（~28GB FP16）
- 原候选中的 40GB 触顶模型（Qwen-Image-Edit-2511、HiDream-I1-Full、CogVideoX1.5-5B）因需独占 A100、部署成本过高，已在筛选阶段移除
- 75% 的模型显存需求 ≤ 8GB，可多服务共卡部署
- 量化版本（INT8/INT4/FP8/FP4）可进一步降低显存，允许同卡并行更多服务

### 2.3 推荐 GPU 配置

| GPU 型号 | 显存 | 适用场景 |
|----------|------|----------|
| NVIDIA A100 40GB | 40GB | 满足所有模型单卡部署 |
| NVIDIA A10 | 24GB | 覆盖 ~90% 模型，大模型需量化 |
| NVIDIA RTX 4090 | 24GB | 性价比选择，覆盖 ~90% 模型 |
| NVIDIA L40S | 48GB | 超出约束上限，可同时跑多个服务 |

---

## 3. 内网部署实用性分析与筛选（120 → 100）

### 3.1 评估维度

对 120 个候选模型按以下 6 个维度评估企业内网部署实用性：

| 维度 | 说明 | 对应的移除标准 |
|------|------|----------------|
| **显存成本** | 显存需求是否接近 40GB 上限，是否需独占 A100 | 触顶显存（~40GB），部署成本过高 |
| **场景通用性** | 是否匹配企业内网常见场景（文档、语音、图像、检索、安全） | 场景过于专用（工业部件设计、音频驱动视频等） |
| **许可合规** | 商用许可是否明确、有无地域限制 | 许可证待确认、存在地域/商业限制 |
| **生态成熟度** | 是否官方维护、集成主流框架、有部署案例 | 社区实验项目、多组件依赖部署复杂 |
| **版本状态** | 是否正式发布版本 | preview / exp 开发版 |
| **冗余度** | 同家族/同架构多版本是否重叠 | 第一代被第二代完全取代、同架构重复 |

### 3.2 各类别内网实用性评估

| 类别 | 候选数 | 保留数 | 移除数 | 内网实用性评估 |
|------|--------|--------|--------|----------------|
| OCR / 文档解析 | 26 | 23 | 3 | **极高**：合同/票据/档案电子化为内网核心场景，保留高精度与轻量两级梯队 |
| 嵌入检索 / 重排序 | 19 | 17 | 2 | **极高**：RAG/知识库检索刚需，仅移除英文专用版本（中文场景实用性弱） |
| 图像分割 / 目标检测 | 8 | 8 | 0 | **高**：SAM3、RMBG、RF-DETR 均为内网图像处理实用模型，全部保留 |
| 语音识别（ASR） | 10 | 9 | 1 | **高**：会议转写/客服质检核心场景，仅移除通用多模态基座模型 |
| 语音合成（TTS） | 15 | 13 | 2 | **高**：语音播报/呼叫中心场景，仅移除 2 个通用音频基座模型 |
| 代码生成 | 3 | 3 | 0 | **中**：辅助编程通用需求，全部保留（部署时注意 30B/16B 显存较高） |
| 其他专用 | 7 | 7 | 0 | **中**：时序预测、隐私过滤、代码安全检测均为内网实用场景，全部保留 |
| 3D 生成 / 重建 | 9 | 6 | 3 | **中低**：电商 3D 展示等场景可用，移除多组件部署复杂及过专模型 |
| 图像生成 / 编辑 | 13 | 9 | 4 | **中低**：营销素材场景可用，移除 2 个触顶显存 + 2 个冗余版本 |
| 视频生成 / 编辑 | 6 | 3 | 3 | **低**：内网场景有限且推理成本高，仅保留轻量/新一代模型 |
| 机器人 / 具身智能 | 4 | 2 | 2 | **低**：需实体机器人硬件，多数企业内网无此场景，仅保留 NVIDIA 官方商用许可版本 |

### 3.3 移除的 20 个模型及原因

| 原序号 | 模型 | 类别 | 移除原因 | 类型 |
|--------|------|------|----------|------|
| 4 | deepseek-ai/DeepSeek-OCR | OCR | 第一代模型，被 DeepSeek-OCR-2 完全取代，功能重叠 | 冗余 |
| 18 | docling-project/SmolDocling-256M-preview | OCR | 名称含 preview，预览版非正式发布，不符合"非开发版"标准 | 开发版 |
| 20 | nanonets/Nanonets-OCR2-1.5B-exp | OCR | 名称含 exp，实验版，生产稳定性无保障 | 开发版 |
| 33 | microsoft/Phi-4-multimodal-instruct | ASR | 通用多模态基座模型，非专用 ASR；12GB 部署成本高，专用 ASR 有更轻替代 | 通用基座 |
| 44 | moonshotai/Kimi-Audio-7B-Instruct | TTS | 音频基座模型（ASR+TTS+对话三位一体），通用性过强，20GB 显存成本高 | 通用基座 |
| 45 | Qwen/Qwen2.5-Omni-7B | TTS | 统一多模态基座模型，16GB 显存，专用 TTS 场景有更轻量替代 | 通用基座 |
| 54 | Qwen/Qwen-Image-Edit-2511 | 图像生成 | ~40GB FP16 触顶 40GB 约束，需独占 A100，内网部署成本过高 | 触顶显存 |
| 57 | stabilityai/stable-diffusion-3.5-large | 图像生成 | 与 large-turbo 同架构完全重复，turbo 4 步蒸馏推理更实用 | 冗余 |
| 61 | black-forest-labs/FLUX.1-schnell | 图像生成 | 老一代 FLUX.1 系列，与 FLUX.2-klein 系列功能重叠，24GB 偏重 | 冗余 |
| 63 | HiDream-ai/HiDream-I1-Full | 图像生成 | ~40GB FP16 触顶，需独占 A100 | 触顶显存 |
| 67 | tencent/HunyuanVideo | 视频生成 | 第一代，被 HunyuanVideo-1.5（8.3B，SSTA 加速）取代 | 冗余 |
| 68 | Skywork/SkyReels-V3-A2V-19B | 视频生成 | 音频驱动+多参考图，场景过于专用；Skywork 社区许可含地域限制 | 场景专用 |
| 70 | zai-org/CogVideoX1.5-5B | 视频生成 | ~40GB FP16 触顶，需独占 A100 | 触顶显存 |
| 74 | tencent/Hunyuan3D-Omni | 3D 生成 | 多组件架构，部署依赖复杂，单服务难以独立交付 | 部署复杂 |
| 75 | Stable-X/Hi3DGen | 3D 生成 | 多组件架构（法向桥接多模型串联），部署链路复杂 | 部署复杂 |
| 77 | tencent/Hunyuan3D-Part | 3D 生成 | 组件级 3D 生成（工业部件设计），内网场景过于专用 | 场景专用 |
| 96 | ibm-granite/granite-embedding-english-r2 | 嵌入 | 英文专用嵌入，内网中文场景实用性弱（同系列 multilingual 版本已保留） | 语言限制 |
| 97 | ibm-granite/granite-embedding-reranker-english-r2 | 重排序 | 英文专用重排序，同上 | 语言限制 |
| 109 | lerobot/smolvla_base | 机器人 | 社区级 VLA 实验项目，需实体机器人硬件，企业内网难落地 | 硬件依赖 |
| 110 | lerobot/pi0_base | 机器人 | 研究性 VLA 模型，需专用机器人平台支撑 | 硬件依赖 |

### 3.4 筛选结论

- **移除 20 个，保留 100 个**，移除理由集中在：触顶显存（3）、冗余重复（4）、通用基座模型（3）、开发/实验版（2）、部署复杂（2）、场景专用（2）、语言限制（2）、硬件依赖（2）
- **23 个高优先级模型全部保留**，核心能力（OCR/ASR/TTS/嵌入/检测）未受影响
- 保留的 100 个模型**全部 ≤ 32GB 显存**，无 40GB 触顶模型，普通 24GB 显卡可覆盖 ~90% 模型
- 机器人/视频生成两类保留数量最少（各 2/3 个），仅保留有明确内网落地场景或官方商用支持的版本

---

## 4. 精选模型清单（100个）

### 4.1 OCR / 文档解析（23个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 1 | baidu/Unlimited-OCR | 文档票据表格图片转可编辑文本和结构化JSON | 3B | ~6 | 高 | 原始保留 |
| 2 | ATH-MaaS/OvisOCR2 | 复杂排版文档手写体多语言OCR识别 | 0.9B | ~1.8 | 高 | 原始保留 |
| 3 | deepseek-ai/DeepSeek-OCR-2 | DeepSeek OCR第二代高精度文档识别 | 3B | ~6 | 高 | 原始保留 |
| 4 | PaddlePaddle/PaddleOCR-VL-1.6 | 多场景端到端OCR视觉语言模型 | 1.0B | ~2 | 有条件 | 原始保留 |
| 5 | PaddlePaddle/PP-OCRv5 | 超轻量级OCR中英文识别边缘设备部署 | 9.7M | ~0.02 | 有条件 | 原始保留 |
| 6 | tencent/HunyuanOCR | 复杂文档表格手写体OCR识别 | 1B | ~2 | 有条件 | 原始保留 |
| 7 | numind/NuMarkdown-8B-Thinking | 文档图片转Markdown保留排版 | 8B | ~16 | 有条件 | 原始保留 |
| 8 | lightonai/LightOnOCR-2-1B | LightOn OCR多语言文档识别 | 1B | ~2 | 有条件 | 原始保留 |
| 9 | nvidia/NVIDIA-Nemotron-Parse-v1.2 | NVIDIA文档解析提取结构化信息 | 0.9B | ~1.8 | 有条件 | 原始保留 |
| 10 | ibm-granite/granite-docling-258M | IBM文档理解PDF转结构化数据 | 0.3B | ~0.6 | 有条件 | 原始保留 |
| 11 | openbmb/MiniCPM-V-4.6 | MiniCPM-V轻量级视觉语言OCR图像理解 | 1B | ~2 | 有条件 | 原始保留 |
| 12 | numind/NuExtract3 | 文档信息抽取结构化实体关系 | 5B | ~10 | 有条件 | 原始保留 |
| 13 | microsoft/Fara1.5-9B | 微软Fara 9B文档理解和OCR | 9B | ~18 | 有条件 | 原始保留 |
| 14 | microsoft/Fara1.5-4B | 微软Fara视觉语言图像理解 | 5B | ~10 | 有条件 | 原始保留 |
| 15 | rednote-hilab/dots.ocr | 多语言文档版面解析转Markdown | 1.7B | ~4 | 新增 | 第一轮补充 |
| 16 | allenai/olmOCR-2-7B-1025 | PDF/扫描件转干净Markdown保留阅读顺序 | 8B | ~16 | 新增 | 第一轮补充 |
| 17 | opendatalab/MinerU2.5-Pro-2604-1.2B | 高分辨率文档PDF转Markdown+JSON | ~1.2B | ~6 | 新增 | 第一轮补充 |
| 18 | stepfun-ai/GOT-OCR2_0 | 端到端通用格式化细粒度OCR | ~580M | ~2 | 新增 | 第一轮补充 |
| 19 | Qwen/Qwen3-VL-8B-Instruct | 通用VLM强文档理解OCR能力 | 8B | ~16 | 新增 | 第一轮补充 |
| 20 | OpenGVLab/InternVL3-2B-Instruct | 通用VLM强文档图表理解 | 2B | ~5 | 新增 | 第一轮补充 |
| 21 | THUDM/GLM-4.1V-9B-Thinking | 视觉语言模型含文档理解思维链 | 9B | ~18 | 新增 | 第一轮补充 |
| 22 | HuggingFaceTB/SmolVLM-Instruct | 超轻量VLM文档截图VQA与OCR | 2B | ~5 | 新增 | 第一轮补充 |
| 23 | zai-org/GLM-OCR | 轻量级多模态OCR复杂文档高精度识别 | 0.9B | ~3 | 新增 | 第二轮补充 |

### 4.2 语音识别 ASR（9个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 24 | nvidia/nemotron-3.5-asr-streaming-0.6b | 语音音频流实时转写流式ASR | 0.6B | ~1.2 | 高 | 原始保留 |
| 25 | pyannote/speaker-diarization-community-1 | 音频说话人分离区分时间段 | ~100M | ~0.2 | 高 | 原始保留 |
| 26 | Qwen/Qwen3-ASR-1.7B | 通义千问语音识别高精度ASR | 2B | ~4 | 有条件 | 原始保留 |
| 27 | nvidia/parakeet-tdt-0.6b-v3 | NVIDIA Parakeet流式ASR | 0.6B | ~1.2 | 有条件 | 原始保留 |
| 28 | Qwen/Qwen3-ASR-0.6B-hf | 通义千问0.6B轻量级语音识别 | 0.8B | ~1.6 | 有条件 | 原始保留 |
| 29 | nvidia/canary-qwen-2.5b | 英语ASR+翻译+语音理解 | 2.5B | ~6 | 新增 | 第一轮补充 |
| 30 | stepfun-ai/Step-Audio-2-mini | 端到端语音对话语音理解+生成 | ~B级 | ~4 | 新增 | 第一轮补充 |
| 31 | CohereLabs/cohere-transcribe-03-2026 | 多语言ASR 14种语言FastConformer | 2B | ~5 | 新增 | 第二轮补充 |
| 32 | OpenMOSS-Team/MOSS-Transcribe-Diarize-0.9B | ASR+说话人分离90分钟长音频 | 0.9B | ~3 | 新增 | 第二轮补充 |

### 4.3 语音合成 TTS（13个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 33 | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 文本生成自然语音定制化音色克隆 | 1.7B | ~3.4 | 高 | 原始保留 |
| 34 | k2-fsa/OmniVoice | 多语言零样本语音合成音色克隆 | 0.6B | ~1.2 | 有条件 | 原始保留 |
| 35 | neuphonic/neutts-2e | 神经TTS高质量自然语音生成 | 0.2B | ~0.4 | 有条件 | 原始保留 |
| 36 | fishaudio/s2-pro | Fish Audio专业版TTS语音合成克隆 | 5B | ~10 | 有条件 | 原始保留 |
| 37 | microsoft/VibeVoice-Realtime-0.5B | 微软实时语音合成低延迟 | 0.5B | ~1 | 有条件 | 原始保留 |
| 38 | microsoft/VibeVoice-1.5B | 微软VibeVoice语音合成 | 3B | ~6 | 有条件 | 原始保留 |
| 39 | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 阿里CosyVoice3语音合成轻量版 | 0.5B | ~1 | 有条件 | 原始保留 |
| 40 | IndexTeam/IndexTTS2 | 零样本语音克隆TTS情感解耦 | ~0.5B | ~4 | 新增 | 第一轮补充 |
| 41 | OpenMOSS-Team/MOSS-TTS | 多语言TTS 20语言低延迟 | ~B级 | ~4 | 新增 | 第一轮补充 |
| 42 | sesame/csm-1b | 对话式语音生成TTS | 1B | ~3 | 新增 | 第一轮补充 |
| 43 | canopylabs/orpheus-3b-0.1-ft | 高情感表现力TTS情感标签 | 3B | ~6 | 新增 | 第一轮补充 |
| 44 | Zyphra/Zonos | 多语言TTS即时语音克隆 | 1.6B | ~4 | 新增 | 第一轮补充 |
| 45 | openbmb/VoxCPM2 | 无分词器多语言TTS 30种语言48kHz | 2B | ~8 | 新增 | 第二轮补充 |

### 4.4 图像生成 / 编辑（9个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 46 | ideogram-ai/ideogram-4-fp8 | Ideogram4 FP8高质量文字渲染图像 | 9B | ~9 | 高 | 原始保留 |
| 47 | black-forest-labs/FLUX.2-klein-9B | 高质量图像生成9B平衡画质速度 | 9B | ~18 | 有条件 | 原始保留 |
| 48 | ideogram-ai/ideogram-4-nf4 | Ideogram4 NF4量化图像生成 | 5B | ~2.5 | 有条件 | 原始保留 |
| 49 | black-forest-labs/FLUX.2-klein-4B | FLUX2 Klein 4B轻量图像生成 | 4B | ~8 | 有条件 | 原始保留 |
| 50 | stabilityai/stable-diffusion-3.5-large-turbo | 4步蒸馏快速文生图 | 8B | ~16 | 新增 | 第一轮补充 |
| 51 | stabilityai/stable-diffusion-3.5-medium | 轻量文生图消费级显卡友好 | 2.5B | ~9 | 新增 | 第一轮补充 |
| 52 | Efficient-Large-Model/Sana_Sprint_1.6B_1024px | 一步高速文生图1024px多比例 | 1.6B | ~9 | 新增 | 第一轮补充 |
| 53 | Efficient-Large-Model/Sana_1.5 | 高效高分辨率文生图线性DiT | 4.8B | ~16 | 新增 | 第一轮补充 |
| 54 | stepfun-ai/Step1X-Edit | 自然语言指令驱动图像编辑 | ~12B | ~28 | 新增 | 第二轮补充 |

### 4.5 视频生成 / 编辑（3个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 55 | Qualcomm-AI-Research/mobilewan | 移动端视频生成5B手机运行 | 5B | ~10 | 有条件 | 原始保留 |
| 56 | Wan-AI/Wan2.1-T2V-1.3B | 万相视频生成轻量版文生视频 | 1B | ~2 | 有条件 | 原始保留 |
| 57 | tencent/HunyuanVideo-1.5 | 文图生视频轻量8.3B SSTA加速 | 8.3B | ~14 | 新增 | 第一轮补充 |

### 4.6 3D 生成 / 重建（6个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 58 | nvidia/instant-nurec | 即时神经场景重建3D | ~1B | ~2 | 高 | 原始保留 |
| 59 | microsoft/TRELLIS.2-4B | 单张图片生成3D模型三维重建 | 4B | ~8 | 有条件 | 原始保留 |
| 60 | tencent/Hunyuan3D-2.1 | 腾讯混元3D图片生成3D模型 | ~3B | ~6 | 有条件 | 原始保留 |
| 61 | stabilityai/stable-fast-3d | 单图转UV纹理网格含PBR材质 | 1.01B | ~6 | 新增 | 第一轮补充 |
| 62 | VAST-AI/TripoSG | 单图转高保真3D网格rectified flow | 1B | ~6 | 新增 | 第二轮补充 |
| 63 | stepfun-ai/Step1X-3D | 高保真可控单图转3D几何+纹理 | 1.3B | ~18 | 新增 | 第二轮补充 |

### 4.7 嵌入检索 / 重排序（17个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 64 | nvidia/Nemotron-3-Embed-1B-BF16 | 文本向量化编码语义搜索推荐 | 1B | ~2 | 高 | 原始保留 |
| 65 | nvidia/Nemotron-3-Embed-8B-BF16 | 8B文本嵌入高精度语义检索RAG | 8B | ~16 | 高 | 原始保留 |
| 66 | nvidia/Nemotron-3-Embed-1B-NVFP4 | 轻量级嵌入FP4量化极速推理 | 0.8B | ~0.4 | 高 | 原始保留 |
| 67 | Qwen/Qwen3-Embedding-8B | 通义千问8B文本嵌入语义检索 | 8B | ~16 | 高 | 原始保留 |
| 68 | Qwen/Qwen3-Embedding-0.6B | 通义千问轻量级嵌入快速编码 | 0.6B | ~1.2 | 高 | 原始保留 |
| 69 | jinaai/jina-embeddings-v5-omni-small | Jina多模态嵌入文本图像统一编码 | 2B | ~4 | 高 | 原始保留 |
| 70 | sentence-transformers/all-MiniLM-L6-v2 | 经典轻量级句子嵌入文本相似度 | 22.7M | ~0.05 | 高 | 原始保留 |
| 71 | BAAI/bge-m3 | 智源BGE多语言嵌入多粒度检索 | ~568M | ~1.1 | 高 | 原始保留 |
| 72 | BAAI/bge-reranker-v2-m3 | 智源BGE重排序搜索结果排序 | 0.6B | ~1.2 | 高 | 原始保留 |
| 73 | google/embeddinggemma-300m | Google Gemma嵌入300M轻量级 | 0.3B | ~0.6 | 高 | 原始保留 |
| 74 | nomic-ai/nomic-embed-text-v2-moe | 多语言MoE文本嵌入101种语言检索 | 305M | ~1 | 新增 | 第一轮补充 |
| 75 | Snowflake/snowflake-arctic-embed-l-v2.0 | 多语言文本检索嵌入RAG优化 | 335M | ~2 | 新增 | 第一轮补充 |
| 76 | Snowflake/snowflake-arctic-embed-m-v2.0 | 中型多语言文本检索嵌入 | 110M | ~1 | 新增 | 第一轮补充 |
| 77 | Qwen/Qwen3-Reranker-0.6B | 多语言文档重排序RAG精排 | 596M | ~2 | 新增 | 第一轮补充 |
| 78 | Qwen/Qwen3-Reranker-4B | 高性能多语言文档重排序旗舰版 | 4B | ~10 | 新增 | 第一轮补充 |
| 79 | Qwen/Qwen3-Embedding-4B | 通用文本嵌入多语言检索分类聚类 | 4B | ~10 | 新增 | 第一轮补充 |
| 80 | ibm-granite/granite-embedding-97m-multilingual-r2 | 多语言文本嵌入200+语言32K上下文 | 97M | ~0.4 | 新增 | 第二轮补充 |

### 4.8 图像分割 / 目标检测（8个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 81 | facebook/sam3 | Meta SAM3通用图像分割一键分割 | 0.9B | ~1.8 | 高 | 原始保留 |
| 82 | facebook/sam3.1 | SAM3.1升级版更精准物体分割 | 0.9B | ~1.8 | 高 | 原始保留 |
| 83 | briaai/RMBG-2.0 | 专业背景移除一键抠图 | 0.2B | ~0.4 | 高 | 原始保留 |
| 84 | facebook/EdgeTAM | 端侧实时视频分割与目标跟踪 | 150M | ~2 | 新增 | 第一轮补充 |
| 85 | Roboflow/rf-detr | 实时目标检测Transformer检测模型 | 200M | ~2 | 新增 | 第一轮补充 |
| 86 | Roboflow/rf-detr-segmentation | 端到端实例分割Deformable DETR | 200M | ~2 | 新增 | 第一轮补充 |
| 87 | IDEA-Research/grounding-dino-1.5-pro | 开放词汇目标检测自然语言描述检测 | 700M | ~3 | 新增 | 第一轮补充 |
| 88 | Intellindust/DEIMv2_DINOv3_X_COCO | 实时目标检测DINOv3增强8种尺寸 | 50.3M | ~2 | 新增 | 第二轮补充 |

### 4.9 机器人 / 具身智能（2个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 89 | nvidia/GR00T-N1.7-3B | NVIDIA GR00T人形机器人控制 | 3B | ~6 | 高 | 原始保留 |
| 90 | nvidia/GR00T-N1.7-DROID | 人形机器人VLA灵巧操作策略商用 | 3B | ~10 | 新增 | 第一轮补充 |

### 4.10 代码生成（3个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 91 | Qwen/Qwen2.5-Coder-7B-Instruct-GGUF | 代码生成GGUF量化辅助编程 | 7B | ~3.5 | 有条件 | 原始保留 |
| 92 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 通义千问代码生成30B辅助编程 | 31B | ~31 | 有条件 | 原始保留 |
| 93 | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 轻量代码生成补全338种编程语言 | 16B(MoE) | ~32 | 新增 | 第一轮补充 |

### 4.11 其他专用（7个）

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 优先级 | 来源 |
|------|----------|----------|--------|----------|--------|------|
| 94 | amazon/chronos-2 | 亚马逊时序预测时间序列趋势 | 0.1B | ~0.2 | 高 | 原始保留 |
| 95 | openai/privacy-filter | 隐私信息过滤检测屏蔽敏感信息 | 1B | ~2 | 有条件 | 原始保留 |
| 96 | amazon/chronos-bolt-base | 零样本时序预测T5概率预测 | 205M | ~1 | 新增 | 第一轮补充 |
| 97 | amazon/chronos-bolt-large | 高精度零样本时序预测大型 | 710M | ~2 | 新增 | 第一轮补充 |
| 98 | amazon/chronos-bolt-mini | 超轻量零样本时序预测边缘场景 | 20M | ~1 | 新增 | 第一轮补充 |
| 99 | google/timesfm-2.5-200m-pytorch | 零样本时序预测16K上下文协变量 | 200M | ~2 | 新增 | 第二轮补充 |
| 100 | cisco-ai/Antares-1B | 代码漏洞定位安全检测128K上下文 | 1B | ~6 | 新增 | 第二轮补充 |

---

## 5. 分类统计

### 5.1 按类别统计

| 类别 | 总数 | 原始保留 | 第一轮补充 | 第二轮补充 | 代表模型 |
|------|------|----------|-----------|-----------|---------|
| OCR / 文档解析 | 23 | 14 | 8 | 1 | Unlimited-OCR、OvisOCR2、DeepSeek-OCR-2、olmOCR-2、GLM-OCR |
| 嵌入检索 / 重排序 | 17 | 10 | 6 | 1 | BGE、Nemotron-Embed、Qwen3-Embedding、Granite-Embedding |
| 语音合成（TTS） | 13 | 7 | 5 | 1 | Qwen3-TTS、CosyVoice3、IndexTTS2、VoxCPM2、Zonos |
| 语音识别（ASR） | 9 | 5 | 2 | 2 | Nemotron-ASR、Qwen3-ASR、Canary-Qwen、Cohere-Transcribe |
| 图像生成 / 编辑 | 9 | 4 | 4 | 1 | FLUX.2-klein、SD3.5-turbo、Sana、Step1X-Edit |
| 图像分割 / 目标检测 | 8 | 3 | 4 | 1 | SAM3、RMBG-2.0、EdgeTAM、RF-DETR、DEIMv2 |
| 3D 生成 / 重建 | 6 | 3 | 1 | 2 | TRELLIS.2、Hunyuan3D、TripoSG、Step1X-3D |
| 其他专用 | 7 | 2 | 3 | 2 | Chronos系列、TimesFM、Antares安全检测 |
| 视频生成 / 编辑 | 3 | 2 | 1 | 0 | Wan2.1、HunyuanVideo-1.5、MobileWan |
| 代码生成 | 3 | 2 | 1 | 0 | Qwen-Coder、DeepSeek-Coder-V2-Lite |
| 机器人 / 具身智能 | 2 | 1 | 1 | 0 | GR00T-N1.7 |
| **合计** | **100** | **53** | **36** | **11** | |

### 5.2 按优先级统计

| 优先级 | 数量 | 占比 |
|--------|------|------|
| 高优先级 | 23 | 23% |
| 有条件可用 | 30 | 30% |
| 新增补充 | 47 | 47% |
| **合计** | **100** | 100% |

> 高优先级 23 个模型全部保留，未受内网实用性筛选影响。

### 5.3 显存分布

| 显存区间 | 模型数量 | 占比 |
|----------|----------|------|
| ≤ 2GB | 44 | 44% |
| 2~8GB | 31 | 31% |
| 8~16GB | 18 | 18% |
| 16~24GB | 4 | 4% |
| 24~32GB | 3 | 3% |
| 32~40GB | 0 | 0% |

> 75% 的模型显存需求 ≤ 8GB，可在单张 24GB 显卡上并行部署多个服务；无 40GB 触顶模型。

### 5.4 来源结构

| 来源 | 数量 | 说明 |
|------|------|------|
| 原始保留（高优先级+有条件可用） | 53 | 从 HF Trending 100 个中移除 44 个后保留 56 个，再移除 3 个后剩余 |
| 第一轮补充 | 36 | 2025-2026 年生产级模型 |
| 第二轮补充 | 11 | 企业级 OCR/ASR/TTS/图像/3D/嵌入/检测/安全模型 |
| **最终合计** | **100** | 从 120 个候选中按内网实用性筛选保留 |

---

## 6. 基础镜像方案

### 6.1 镜像分层设计

| 层 | 内容 | 是否固化 | 体积估算 |
|----|------|----------|----------|
| **L1 系统层** | CUDA 12.1 + cuDNN + Python 3.11 + ffmpeg + 中文字体 + tesseract + 构建工具 | 固化 | ~4GB |
| **L2 框架层** | PyTorch 2.4 (cu121) + transformers + accelerate + huggingface_hub + safetensors + bitsandbytes + onnxruntime-gpu | 固化 | ~5GB |
| **L3 专用依赖** | paddleocr / diffusers / funasr / gguf 等按模型类别 | **不固化**，运行时装 | 动态 |
| **L4 业务文件** | requirements / 脚本 / 配置 / 业务代码 | **不固化**，宿主机挂载 | 0 |
| **L5 模型权重** | 模型权重、HF 缓存 | **不固化**，持久卷挂载 | 0 |

镜像总体积约 **9GB**，构建一次后稳定复用。

### 6.2 镜像内容定义

```
镜像内：
├── /usr/local/cuda          ← CUDA 12.1 + cuDNN（来自 nvidia/cuda 基础镜像）
├── /usr/bin/python          ← Python 3.11
├── site-packages/
│   ├── torch 2.4            ← PyTorch (cu121)
│   ├── transformers
│   ├── accelerate
│   ├── huggingface_hub
│   ├── safetensors
│   ├── bitsandbytes
│   ├── onnxruntime-gpu
│   └── (通用基础库)
├── /usr/bin/ffmpeg          ← 多媒体处理
├── /usr/bin/tesseract       ← OCR 引擎
└── /usr/share/fonts/        ← 中文字体 (Noto CJK)

镜像内【没有】：
×  requirements-*.txt
×  entrypoint.sh / 任何脚本
×  模型权重
×  业务代码
×  ENTRYPOINT / CMD
```

### 6.3 宿主机目录结构（所有动态内容）

```
/host/deploy/                        # 挂载到容器 /deploy
├── requirements/                    # 按类别的 pip 依赖清单
│   ├── ocr.txt                      # paddleocr, rapidocr, surya, pymupdf ...
│   ├── asr.txt                      # funasr, faster-whisper ...
│   ├── tts.txt                      # cosyvoice, fish-speech ...
│   ├── image-gen.txt                # diffusers, compel ...
│   ├── video-gen.txt                # diffusers[video] ...
│   ├── embedding.txt                # sentence-transformers ...
│   ├── robot.txt                    # (机器人相关)
│   ├── 3d-gen.txt                   # trellis, hunyuan3d ...
│   ├── code-gen.txt                 # llama-cpp-python (GGUF 模型) ...
│   └── common.txt                   # 通用补充依赖
├── scripts/
│   ├── entrypoint.sh               # 启动入口：装包→加载模型→启动服务
│   ├── download_models.py          # 按需拉取权重到 /data/models
│   └── healthcheck.py              # 健康检查
├── config/
│   └── models.yaml                 # 模型ID→类别→端口→显存 映射
└── app/
    └── server.py                   # 业务服务代码

持久卷：
├── /data/models                    # 模型权重 + HF 缓存（所有服务共享）
├── /data/pip-cache                 # pip 缓存（所有服务共享，加速装包）
└── /data/outputs                   # 推理输出
```

**修改任何 requirements、脚本或配置后，重启容器即生效，镜像无需重建。**

### 6.4 环境变量设计

镜像内预设的环境变量（指向挂载路径）：

```dockerfile
ENV HF_HOME=/data/models/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/data/models/hf_cache/hub
ENV TRANSFORMERS_CACHE=/data/models/hf_cache/hub
ENV PIP_CACHE_DIR=/data/pip-cache
ENV TORCH_HOME=/data/models/torch_cache
ENV MODELS_ROOT=/data/models
ENV OUTPUTS_ROOT=/data/outputs
```

运行时注入的环境变量（docker run / compose）：

| 变量 | 作用 | 示例值 |
|------|------|--------|
| `MODEL_TYPE` | 决定装哪类 pip 包 | `ocr` / `tts` / `image-gen` |
| `MODEL_ID` | 决定加载哪个模型权重 | `baidu/Unlimited-OCR` |
| `SERVICE_PORT` | 服务监听端口 | `8000` |
| `NVIDIA_VISIBLE_DEVICES` | 指定 GPU | `0` / `0,1` |
| `HF_ENDPOINT` | HF 镜像源（国内） | `https://hf-mirror.com` |

### 6.5 部署编排示例

```yaml
# docker-compose.yml — 单镜像多服务
services:
  ocr-service:
    image: model-deploy:1.0             # 同一个镜像，永不重建
    command: ["/deploy/scripts/entrypoint.sh"]
    environment:
      - MODEL_TYPE=ocr
      - MODEL_ID=baidu/Unlimited-OCR
      - SERVICE_PORT=8000
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - /host/deploy:/deploy            # 脚本 + requirements + 配置
      - model-store:/data/models         # 权重（所有服务共享）
      - pip-cache:/data/pip-cache        # pip 缓存（所有服务共享）
      - /host/outputs:/data/outputs
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  tts-service:
    image: model-deploy:1.0             # 同一个镜像
    command: ["/deploy/scripts/entrypoint.sh"]
    environment:
      - MODEL_TYPE=tts
      - MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
      - SERVICE_PORT=8001
      - NVIDIA_VISIBLE_DEVICES=0         # 同卡共部署（显存足够）
    volumes:
      - /host/deploy:/deploy
      - model-store:/data/models
      - pip-cache:/data/pip-cache
      - /host/outputs:/data/outputs
    ports:
      - "8001:8001"

volumes:
  model-store:
  pip-cache:
```

### 6.6 entrypoint.sh 职责

该脚本挂载在 `/deploy/scripts/entrypoint.sh`，镜像启动时由外部 command 指定执行：

```
1. 读取 MODEL_TYPE 环境变量
2. pip install -r /deploy/requirements/${MODEL_TYPE}.txt
   （命中 /data/pip-cache 则秒级完成，首次约 2~5 分钟）
3. 读取 MODEL_ID 环境变量
4. 检查 /data/models 中是否已有该模型权重
   - 无 → 调用 download_models.py 下载
   - 有 → 直接跳过
5. 执行 /deploy/app/server.py 拉起服务
```

### 6.7 冷启动优化

| 场景 | 首次启动 | 二次启动（缓存命中） |
|------|----------|---------------------|
| pip 装包 | 2~5 分钟 | < 10 秒 |
| 模型权重下载 | 取决于模型大小和带宽 | 0 秒（本地命中） |
| 服务启动 | 10~30 秒 | 10~30 秒 |

> 生产环境建议提前预热：对每个类别执行一次空跑，让 pip 缓存和权重缓存填充到持久卷。

---

## 7. CI/CD 持续集成方案

### 7.1 推荐项目结构

建议建立两个独立的 Git 仓库（项目），职责分离：

```
仓库1：model-deploy-image          # 镜像构建项目
├── Dockerfile                    # 镜像定义（纯运行时）
├── requirements-base.txt         # 镜像固化的通用框架依赖
├── .dockerignore
├── .gitlab-ci.yml / .github/workflows/build.yml
└── README.md

仓库2：model-deploy-config         # 部署配置项目（高频修改）
├── requirements/
│   ├── ocr.txt
│   ├── asr.txt
│   ├── tts.txt
│   ├── image-gen.txt
│   ├── video-gen.txt
│   ├── embedding.txt
│   ├── robot.txt
│   ├── 3d-gen.txt
│   ├── code-gen.txt
│   └── common.txt
├── scripts/
│   ├── entrypoint.sh
│   ├── download_models.py
│   └── healthcheck.py
├── config/
│   └── models.yaml
├── app/
│   └── server.py
├── docker-compose.yml
├── .gitlab-ci.yml / .github/workflows/deploy.yml
└── README.md
```

**分离原因**：
- 镜像仓库变更频率极低（CUDA/PyTorch 大版本升级才动），CI 触发少
- 配置仓库变更频率高（增减模型、改依赖、改脚本），CI 高频触发但不需重建镜像

### 7.2 镜像构建流水线（仓库1）

触发条件：Dockerfile 或 requirements-base.txt 变更

```yaml
name: Build Base Image
on:
  push:
    paths:
      - 'Dockerfile'
      - 'requirements-base.txt'
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: <your-registry>
          username: ${{ secrets.REG_USER }}
          password: ${{ secrets.REG_PASS }}
      - name: Build and Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            <your-registry>/model-deploy:1.0
            <your-registry>/model-deploy:latest
          cache-from: type=registry,ref=<your-registry>/model-deploy:cache
          cache-to: type=registry,ref=<your-registry>/model-deploy:cache,mode=max
```

### 7.3 部署配置流水线（仓库2）

触发条件：requirements / scripts / config / app 任意变更

```yaml
name: Deploy Config Update
on:
  push:
    paths:
      - 'requirements/**'
      - 'scripts/**'
      - 'config/**'
      - 'app/**'
      - 'docker-compose.yml'
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sync to deploy server
        run: |
          rsync -avz --delete requirements/ scripts/ config/ app/ \
            docker-compose.yml deploy@<server>:/opt/deploy/
      - name: Restart affected services
        run: |
          ssh deploy@<server> "cd /opt/deploy && \
            docker compose up -d --force-recreate"
```

### 7.4 持续修改流程

| 修改场景 | 操作 | 是否重建镜像 | CI 触发 |
|----------|------|-------------|---------|
| 新增一个模型 | 改 config/models.yaml + docker-compose 加服务 | 否 | 配置仓库 CI |
| 修改某类 pip 依赖 | 改 requirements/xxx.txt | 否 | 配置仓库 CI |
| 升级 transformers 版本 | 改 requirements-base.txt | 是 | 镜像仓库 CI |
| 升级 PyTorch 大版本 | 改 Dockerfile + requirements-base.txt | 是 | 镜像仓库 CI |
| 修改启动逻辑 | 改 scripts/entrypoint.sh | 否 | 配置仓库 CI |
| 修改业务代码 | 改 app/server.py | 否 | 配置仓库 CI |

### 7.5 版本管理建议

- 镜像版本：`model-deploy:1.0` → `1.1`（小升级）→ `2.0`（大升级）
- 配置版本：Git 分支管理，main 为生产环境，dev 为测试环境
- 模型权重版本：通过 `MODEL_ID` 环境变量控制，HF 上游版本变更时更新

---

## 8. 注意事项与已知约束

### 8.1 模型兼容性约束

| 问题 | 影响 | 规避方式 |
|------|------|----------|
| PyTorch 版本 | 95% 模型可用 torch 2.4 (cu121)；极少数 2026 新模型可能需 torch 2.5+ | 镜像固化 torch 2.4 为主力；个别模型在 entrypoint 里 `pip install -U torch` 覆盖 |
| PaddlePaddle 自带 CUDA | paddlepaddle-gpu 捆绑 CUDA 库，与镜像 CUDA 共存基本无冲突 | 运行时装即可，无需特殊处理 |
| GGUF 模型（序号 91） | 走 llama-cpp-python 而非 PyTorch | 在 code-gen.txt 中加 llama-cpp-python |
| bitsandbytes 版本匹配 | 必须与 torch/CUDA 版本严格匹配 | 镜像固化匹配版本；若 entrypoint 升级 torch 则同步重装 bitsandbytes |
| flash-attn 编译慢 | 体积大、编译 10+ 分钟，仅 Ampere+ 显卡可用 | 默认不装，按需在 requirements 中解锁 |
| 最大显存模型（序号 92/93） | Qwen3-Coder-30B ~31GB、DeepSeek-Coder-V2-Lite ~32GB，接近 40GB 上限 | 使用量化版本（INT8 降至 ~16GB，INT4 降至 ~8GB）或独占大显存显卡 |
| DeepSeek-Coder-V2-Lite（序号 93） | 16B MoE 总参数，FP16 ~32GB | INT4 量化后 ~8GB，部署时建议使用量化版 |
| Step1X-Edit（序号 54） | ~28GB FP16，显存需求较高 | 独占 40GB 显卡或量化部署 |
| 通用 VLM 模型（序号 11、19-22） | 为通用 VLM，因其 OCR/文档理解能力纳入 | 专用任务场景使用，非作为通用聊天机器人 |
| Antares-1B 安全检测（序号 100） | 128K 上下文需较大 KV Cache | 实际显存可能高于 6GB 估算，建议 8GB+ 显卡部署 |

### 8.2 部署运维约束

| 问题 | 说明 |
|------|------|
| 冷启动延迟 | 首次装某类专用包 2~5 分钟；pip 缓存卷持久化后二次启动降至秒级 |
| 权重下载 | 首次拉取大模型（如 30B）可能耗时较长；建议离线预下载到持久卷 |
| 国内网络 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站 |
| 多服务共卡 | 显存 ≤ 8GB 的模型可同卡并行；大模型独占一张卡 |
| 宿主机驱动 | 必须安装 NVIDIA 驱动 + NVIDIA Container Toolkit，否则容器无法访问 GPU |
| 许可证合规 | Stability Community 许可证对年营收 >$1M 企业有商用限制；Tencent/Skywork Community 有地域限制（SkyReels 已移除，注意 Hunyuan 系列） |
| Antares 气隙部署 | Cisco Antares 专为气隙/本地部署设计，源代码不离开信任域，适合内网安全审计 |

### 8.3 显存估算参考

| 精度 | 估算公式 |
|------|----------|
| FP16 / BF16 | 参数量(B) × 2 ≈ 显存(GB) |
| INT8 / FP8 | 约为 FP16 的 50% |
| INT4 / NF4 / FP4 | 约为 FP16 的 25% |
| 1.58-bit (BitNet) | 约为 FP16 的 10% |

> 实际部署还需考虑框架开销、KV Cache 等，以上为模型权重估算值。

### 8.4 已移除模型汇总

**历史筛选（120 候选形成过程）**：从 HF Trending 原始 100 个中共移除 47 个（44 个第一轮 + 3 个第二轮），主要原因分布：

| 移除原因 | 数量 | 典型模型 |
|----------|------|---------|
| 个人/社区项目，生产验证不足 | 16 | Kokoro-82M、Gepard、VoxCPM2（后纠正纳入） |
| 开发版/实验性，非生产优化 | 6 | FLUX.2-dev、FLUX.1-dev、Mage-Flow 系列 |
| 场景过于专用 | 8 | MiniCPM-RobotTrack、Miril-Drone、SCAIL-2 |
| 生态不成熟/商业化不明 | 7 | Krea-2 系列、Z-Image-Turbo、Sulphur-2 |
| 社区项目，企业级支持不足 | 4 | GigaAM、tiron、Chandra-OCR、BitNet-Embedding |
| 第三方未授权优化版 | 3 | OvisOCR2-GGUF、Krea2-Turbo-HD、Flux2-Klein-True-V3 |
| 参数未公开/许可证待确认 | 2 | SkyReels-A2、GR00T-N1.6-3B |
| 专用仿真场景，内网通用性不足 | 1 | asset-harvester |

**本次内网实用性再筛选（120 → 100）**：移除 20 个模型，移除原因分布：

| 移除原因 | 数量 | 典型模型 |
|----------|------|---------|
| 触顶显存（~40GB 需独占 A100） | 3 | Qwen-Image-Edit-2511、HiDream-I1-Full、CogVideoX1.5-5B |
| 冗余重复（被同家族新版本取代） | 4 | DeepSeek-OCR、SD3.5-large、FLUX.1-schnell、HunyuanVideo |
| 通用基座模型（非专用任务） | 3 | Phi-4-multimodal、Kimi-Audio-7B、Qwen2.5-Omni-7B |
| 部署复杂（多组件架构） | 2 | Hunyuan3D-Omni、Hi3DGen |
| 开发版/实验版 | 2 | SmolDocling-preview、Nanonets-OCR2-exp |
| 场景过于专用 | 2 | SkyReels-V3-A2V、Hunyuan3D-Part |
| 英文专用（内网中文场景弱） | 2 | Granite-Embedding-EN、Granite-Reranker-EN |
| 依赖实体机器人硬件 | 2 | smolvla_base、pi0_base |

> 移除明细见 [3.3 移除的 20 个模型及原因](#33-移除的-20-个模型及原因)。

---

## 附录

### A. 手动补充模型说明

以下 4 个模型为按要求手动补充（不在 Trending 筛选范围内），已计入候选模型。本次内网实用性筛选后保留 3 个、移除 1 个：

| 原序号 | 模型 | 类型 | 状态 |
|--------|------|------|------|
| 5 | PaddlePaddle/PaddleOCR-VL-1.6 | OCR | **保留**（现序号 4） |
| 6 | PaddlePaddle/PP-OCRv5 | OCR | **保留**（现序号 5） |
| 7 | tencent/HunyuanOCR | OCR | **保留**（现序号 6） |
| 4 | deepseek-ai/DeepSeek-OCR | OCR | **移除**（被 DeepSeek-OCR-2 完全取代） |
