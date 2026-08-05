# AI 模型部署平台方案文档

> **用途**：本文档作为新建项目的方案基线，用于镜像构建与 CI/CD 持续集成的依据，支持持续修改迭代。
>
> **最后更新**：2026-08-05
>
> **状态**：方案确认中

---

## 目录

1. [项目概述](#1-项目概述)
2. [云服务器资源约束](#2-云服务器资源约束)
3. [精选模型清单](#3-精选模型清单)
4. [基础镜像方案](#4-基础镜像方案)
5. [CI/CD 持续集成方案](#5-cicd-持续集成方案)
6. [注意事项与已知约束](#6-注意事项与已知约束)

---

## 1. 项目概述

### 1.1 目标

基于 Hugging Face Trending 榜单筛选出的 100 个专用 AI 模型，构建**单镜像通用部署平台**，实现：

- 一个镜像覆盖所有模型的部署需求
- 模型权重与业务配置与镜像解耦，放持久化存储
- 专用 pip 依赖在容器启动时按需安装
- 镜像构建一次后稳定复用，支持 CI/CD 持续迭代

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| 镜像最小化 | 镜像只固化二进制运行环境（CUDA + Python + PyTorch + 通用框架），不含任何配置文件、脚本、requirements |
| 权重外挂 | 模型权重放持久化卷，按需加载，不入镜像 |
| 依赖外挂 | 按模型类别的 requirements 文件放宿主机，挂载进容器，运行时安装 |
| 零配置镜像 | 镜像内无 ENTRYPOINT/CMD/业务文件，启动命令完全由外部注入 |
| 单镜像复用 | 所有模型服务共用同一镜像，差异通过环境变量和挂载点区分 |

### 1.3 GPU 驱动说明

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

- 所有 100 个模型的显存需求均在 40GB 以内
- 最大显存需求模型：`Qwen/Qwen-Image-Edit-2511`（~40GB FP16）和 `Qwen/Qwen-Image`（~40GB FP16），刚好触顶
- 大部分模型在 10GB 以内，可多服务共卡部署
- 量化版本（INT8/INT4/FP8/FP4）可进一步降低显存，允许同卡并行更多服务

### 2.3 推荐 GPU 配置

| GPU 型号 | 显存 | 适用场景 |
|----------|------|----------|
| NVIDIA A100 40GB | 40GB | 满足所有模型单卡部署，触顶模型唯一选择 |
| NVIDIA A10 | 24GB | 覆盖 ~90% 模型，触顶模型需量化 |
| NVIDIA RTX 4090 | 24GB | 性价比选择，覆盖 ~90% 模型 |
| NVIDIA L40S | 48GB | 超出约束上限，可同时跑多个服务 |

---

## 3. 精选模型清单

### 3.1 筛选标准

| 条件 | 要求 |
|------|------|
| 发布时间 | 2025年1月1日之后发布或更新 |
| 模型类型 | 仅保留专用任务模型（OCR、ASR、TTS、图像生成、视频生成、嵌入检索、机器人控制等），排除通用 LLM/VLM |
| 部署显存 | 推理所需显存 ≤ 40GB（FP16 估算；量化版本更省） |
| 实用性 | 可直接部署、用于生产环境的模型 |
| 领域限制 | 排除医疗、金融、小说/剧本创作等垂直领域模型 |

### 3.2 分类统计

| 类别 | 数量 | 代表模型 |
|------|------|---------|
| OCR / 文档解析 | 14 | Unlimited-OCR、DeepSeek-OCR、PaddleOCR-VL、PP-OCRv5、HunyuanOCR |
| 语音识别（ASR） | 11 | Nemotron-ASR、Qwen3-ASR、VibeVoice-ASR、Parakeet、Cohere-Transcribe |
| 语音合成（TTS） | 14 | Qwen3-TTS、Kokoro、CosyVoice、VibeVoice、NeuTTS |
| 图像生成 / 编辑 | 17 | FLUX 系列、Mage-Flow、Krea-2、Qwen-Image、Ideogram |
| 视频生成 / 编辑 | 10 | HunyuanVideo、Wan2.x、LTX-2.3、Sulphur-2 |
| 嵌入检索 / 重排序 | 12 | BGE、Nemotron-Embed、Qwen-Embedding、Jina、MiniLM |
| 图像分割 / 目标检测 | 6 | SAM3、RMBG-2.0、DINOv3、Lucida |
| 机器人 / 具身智能 | 7 | MiniCPM-Robot 系列、GR00T、Hy-Embodied-RxBrain |
| 3D 生成 / 重建 | 4 | TRELLIS.2、Hunyuan3D、Instant-NURC |
| 代码生成 | 3 | Qwen-Coder、Veriloop-coder |
| 其他专用 | 12 | 音效生成、时序预测、Prompt 检测、隐私过滤、世界模型 |

### 3.3 完整 100 模型清单

| 序号 | 模型全称 | 核心任务 | 参数量 | 显存(GB) | 类别 |
|------|----------|----------|--------|----------|------|
| 1 | baidu/Unlimited-OCR | 文档票据表格图片转可编辑文本和结构化JSON | 3B | ~6 | OCR |
| 2 | microsoft/Mage-Flow | 文字描述生成产品设计图、营销素材 | 4B | ~8 | 图像生成 |
| 3 | openbmb/MiniCPM-RobotManip | 机械臂抓取放置装配等机器人操作 | 2B | ~4 | 机器人 |
| 4 | ATH-MaaS/OvisOCR2 | 复杂排版文档手写体多语言OCR识别 | 0.9B | ~1.8 | OCR |
| 5 | openbmb/MiniCPM-RobotTrack | 机器人视觉运动目标实时跟踪定位 | 0.4B | ~0.8 | 机器人 |
| 6 | nvidia/nemotron-3.5-asr-streaming-0.6b | 语音音频流实时转写流式ASR | 0.6B | ~1.2 | ASR |
| 7 | OpenMOSS-Team/MOSS-Transcribe-Diarize | 会议录音转写区分说话人生成会议纪要 | 0.9B | ~1.8 | ASR |
| 8 | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | 文本生成自然语音定制化音色克隆 | 1.7B | ~3.4 | TTS |
| 9 | microsoft/Mage-Flow-Edit-Turbo | 图片快速编辑风格转换局部修改 | 4B | ~8 | 图像生成 |
| 10 | krea/Krea-2-Turbo | 快速生成高质量写实风格图片 | 13B | ~26 | 图像生成 |
| 11 | nvidia/Nemotron-3-Embed-1B-BF16 | 文本向量化编码语义搜索推荐 | 1B | ~2 | 嵌入 |
| 12 | Lightricks/LTX-2.3 | 文字描述生成高质量视频片段 | 22B | ~22 | 视频生成 |
| 13 | deepseek-ai/DeepSeek-OCR | 文档票据表格公式高精度OCR结构化输出 | 3B | ~6 | OCR |
| 14 | PaddlePaddle/PaddleOCR-VL-1.6 | 多场景端到端OCR视觉语言模型 | 1.0B | ~2 | OCR |
| 15 | PaddlePaddle/PP-OCRv5 | 超轻量级OCR中英文识别边缘设备部署 | 9.7M | ~0.02 | OCR |
| 16 | tencent/HunyuanOCR | 复杂文档表格手写体OCR识别 | 1B | ~2 | OCR |
| 17 | microsoft/Mage-Flow-Base | 基础版文本生成图像 | 4B | ~8 | 图像生成 |
| 18 | PaddlePaddle/HPD-Parsing | 文档页面结构化解析提取标题段落表格 | 1B | ~2 | OCR |
| 19 | nvidia/Nemotron-3-Embed-8B-BF16 | 8B文本嵌入高精度语义检索RAG | 8B | ~16 | 嵌入 |
| 20 | nvidia/Nemotron-3-Embed-1B-NVFP4 | 轻量级嵌入FP4量化极速推理 | 0.8B | ~0.4 | 嵌入 |
| 21 | egeorcun/lucida | 图像高精度语义分割物体轮廓区域 | 0.2B | ~0.4 | 图像分割 |
| 22 | facebook/sam3 | Meta SAM3通用图像分割一键分割 | 0.9B | ~1.8 | 图像分割 |
| 23 | black-forest-labs/FLUX.2-klein-9B | 高质量图像生成9B平衡画质速度 | 9B | ~18 | 图像生成 |
| 24 | hexgrad/Kokoro-82M | 超轻量级TTS 82M快速语音合成 | 82M | ~0.16 | TTS |
| 25 | pyannote/speaker-diarization-community-1 | 音频说话人分离区分时间段 | ~100M | ~0.2 | ASR |
| 26 | google/tabfm-1.0.0-pytorch | 表格数据分类预测结构化数据处理 | ~500M | ~1 | 其他 |
| 27 | krea/Krea-2-Raw | 原始版高质量图像生成精细控制 | 13B | ~26 | 图像生成 |
| 28 | Qualcomm-AI-Research/mobilewan | 移动端视频生成5B手机运行 | 5B | ~10 | 视频生成 |
| 29 | Tongyi-MAI/Z-Image-Turbo | 快速图像生成Turbo极速出图 | 6B | ~12 | 图像生成 |
| 30 | bosonai/higgs-tts-3-4b | 高质量TTS 4B自然语音生成 | 5B | ~10 | TTS |
| 31 | Qwen/Qwen-Image-Edit-2511 | 图像编辑文字描述修改图片 | 20B | ~40 | 图像生成 |
| 32 | k2-fsa/OmniVoice | 多语言零样本语音合成音色克隆 | 0.6B | ~1.2 | TTS |
| 33 | microsoft/TRELLIS.2-4B | 单张图片生成3D模型三维重建 | 4B | ~8 | 3D生成 |
| 34 | facebook/sam3.1 | SAM3.1升级版更精准物体分割 | 0.9B | ~1.8 | 图像分割 |
| 35 | nineninesix/gepard-1.0 | 轻量级TTS快速自然语音合成 | 0.6B | ~1.2 | TTS |
| 36 | wikeeyang/Krea2-Turbo-HD-V1 | Krea2高清增强版图像生成 | 13B | ~26 | 图像生成 |
| 37 | tencent/Hy-Embodied-RxBrain-1.0 | 具身智能机器人大脑控制机器人 | 6B | ~12 | 机器人 |
| 38 | CohereLabs/cohere-transcribe-arabic-07-2026 | 阿拉伯语语音识别转写 | 2B | ~4 | ASR |
| 39 | ai-sage/GigaAM-Multilingual | 多语言语音识别语音转文字 | ~1B | ~2 | ASR |
| 40 | Trelis/tiron | 高质量语音识别对话式转写 | 2B | ~4 | ASR |
| 41 | facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3图像特征提取检索匹配 | 0.3B | ~0.6 | 图像分割 |
| 42 | black-forest-labs/FLUX.2-dev | FLUX.2开发版最高质量图像生成 | 32B | ~32 | 图像生成 |
| 43 | microsoft/VibeVoice-ASR | 微软VibeVoice高质量语音转文字 | 9B | ~18 | ASR |
| 44 | MCG-NJU/VideoChat3-4B | 视频理解对话分析视频内容 | 4B | ~8 | 其他 |
| 45 | neuphonic/neutts-2e | 神经TTS高质量自然语音生成 | 0.2B | ~0.4 | TTS |
| 46 | CohereLabs/cohere-transcribe-03-2026 | Cohere通用语音识别转写 | 2B | ~4 | ASR |
| 47 | ideogram-ai/ideogram-4-fp8 | Ideogram4 FP8高质量文字渲染图像 | 9B | ~9 | 图像生成 |
| 48 | meta-llama/Prompt-Guard-86M | 轻量级提示词注入检测识别恶意Prompt | 0.3B | ~0.6 | 其他 |
| 49 | fishaudio/s2-pro | Fish Audio专业版TTS语音合成克隆 | 5B | ~10 | TTS |
| 50 | wikeeyang/Flux2-Klein-9B-True-V3 | FLUX2 Klein优化版图像生成 | 9B | ~18 | 图像生成 |
| 51 | numind/NuMarkdown-8B-Thinking | 文档图片转Markdown保留排版 | 8B | ~16 | OCR |
| 52 | SulphurAI/Sulphur-2-base | 基础版文本生成视频 | 9B | ~18 | 视频生成 |
| 53 | microsoft/VibeVoice-Realtime-0.5B | 微软实时语音合成低延迟 | 0.5B | ~1 | TTS |
| 54 | briaai/RMBG-2.0 | 专业背景移除一键抠图 | 0.2B | ~0.4 | 图像分割 |
| 55 | lightonai/LightOnOCR-2-1B | LightOn OCR多语言文档识别 | 1B | ~2 | OCR |
| 56 | Qwen/Qwen3-ASR-1.7B | 通义千问语音识别高精度ASR | 2B | ~4 | ASR |
| 57 | ideogram-ai/ideogram-4-nf4 | Ideogram4 NF4量化图像生成 | 5B | ~2.5 | 图像生成 |
| 58 | Qwen/Qwen3-Embedding-8B | 通义千问8B文本嵌入语义检索 | 8B | ~16 | 嵌入 |
| 59 | Qwen/Qwen-Image | 通义千问图像生成高质量文生图 | 20B | ~40 | 图像生成 |
| 60 | deepseek-ai/DeepSeek-OCR-2 | DeepSeek OCR第二代高精度文档识别 | 3B | ~6 | OCR |
| 61 | openbmb/VoxCPM2 | 开源中文TTS语音合成 | 2B | ~4 | TTS |
| 62 | nvidia/GR00T-N1.7-3B | NVIDIA GR00T人形机器人控制 | 3B | ~6 | 机器人 |
| 63 | nvidia/nemotron-labs-audio-visual-flamingo-hf | 音视频多模态理解分析 | 9B | ~18 | 其他 |
| 64 | nvidia/instant-nurec | 即时神经场景重建3D | ~1B | ~2 | 3D生成 |
| 65 | zai-org/SCAIL-2 | 图像转视频静态图片转动态视频 | ~5B | ~10 | 视频生成 |
| 66 | MirilAI/Miril-Drone-2B-1 | 无人机视觉理解导航目标识别 | 5B | ~10 | 机器人 |
| 67 | Abiray/OvisOCR2-GGUF | OvisOCR2 GGUF量化轻量部署 | 0.8B | ~0.8 | OCR |
| 68 | Qwen/Qwen2.5-Coder-7B-Instruct-GGUF | 代码生成GGUF量化辅助编程 | 7B | ~3.5 | 代码生成 |
| 69 | microsoft/Fara1.5-4B | 微软Fara视觉语言图像理解 | 5B | ~10 | 其他 |
| 70 | Wan-AI/Wan2.1-T2V-1.3B | 万相视频生成轻量版文生视频 | 1B | ~2 | 视频生成 |
| 71 | ibm-granite/granite-docling-258M | IBM文档理解PDF转结构化数据 | 0.3B | ~0.6 | OCR |
| 72 | Qwen/Qwen3-Embedding-0.6B | 通义千问轻量级嵌入快速编码 | 0.6B | ~1.2 | 嵌入 |
| 73 | tencent/Hunyuan3D-2.1 | 腾讯混元3D图片生成3D模型 | ~3B | ~6 | 3D生成 |
| 74 | Wan-AI/Wan2.2-TI2V-5B | 万相2.2图文生视频 | 5B | ~10 | 视频生成 |
| 75 | microsoft/VibeVoice-1.5B | 微软VibeVoice语音合成 | 3B | ~6 | TTS |
| 76 | amazon/chronos-2 | 亚马逊时序预测时间序列趋势 | 0.1B | ~0.2 | 其他 |
| 77 | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 阿里CosyVoice3语音合成轻量版 | 0.5B | ~1 | TTS |
| 78 | nvidia/NVIDIA-Nemotron-Parse-v1.2 | NVIDIA文档解析提取结构化信息 | 0.9B | ~1.8 | OCR |
| 79 | tsinghua-sigs-robot-lab/veriloop-coder-e1 | 清华机器人代码生成控制代码 | 28B | ~28 | 代码生成 |
| 80 | jinaai/jina-embeddings-v5-omni-small | Jina多模态嵌入文本图像统一编码 | 2B | ~4 | 嵌入 |
| 81 | openbmb/MiniCPM-V-4.6 | MiniCPM-V轻量级视觉语言OCR图像理解 | 1B | ~2 | OCR |
| 82 | ResembleAI/chatterbox-nano | 纳米级TTS超轻量语音合成 | ~100M | ~0.2 | TTS |
| 83 | tencent/HunyuanVideo | 腾讯混元视频生成高质量文生视频 | ~12B | ~24 | 视频生成 |
| 84 | nvidia/parakeet-tdt-0.6b-v3 | NVIDIA Parakeet流式ASR | 0.6B | ~1.2 | ASR |
| 85 | black-forest-labs/FLUX.2-klein-4B | FLUX2 Klein 4B轻量图像生成 | 4B | ~8 | 图像生成 |
| 86 | sentence-transformers/all-MiniLM-L6-v2 | 经典轻量级句子嵌入文本相似度 | 22.7M | ~0.05 | 嵌入 |
| 87 | BAAI/bge-m3 | 智源BGE多语言嵌入多粒度检索 | ~568M | ~1.1 | 嵌入 |
| 88 | BAAI/bge-reranker-v2-m3 | 智源BGE重排序搜索结果排序 | 0.6B | ~1.2 | 嵌入 |
| 89 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 通义千问代码生成辅助编程 | 31B | ~31 | 代码生成 |
| 90 | google/embeddinggemma-300m | Google Gemma嵌入300M轻量级 | 0.3B | ~0.6 | 嵌入 |
| 91 | Qwen/Qwen3-ASR-0.6B-hf | 通义千问0.6B轻量级语音识别 | 0.8B | ~1.6 | ASR |
| 92 | OpenMOSS-Team/MOSS-SoundEffect-v2.0 | 音效生成根据描述生成音效 | 1B | ~2 | 其他 |
| 93 | Lightricks/LTX-2.3-22b-IC-LoRA-Foley-V2A | 视频配音效自动生成音效配乐 | ~1B | ~2 | 其他 |
| 94 | datalab-to/chandra-ocr-2 | Chandra OCR文档图像文字识别 | 5B | ~10 | OCR |
| 95 | openai/privacy-filter | 隐私信息过滤检测屏蔽敏感信息 | 1B | ~2 | 其他 |
| 96 | acvlab/ABot-World-0-5B-LF | 机器人世界模型预测动作结果 | 5B | ~10 | 机器人 |
| 97 | microsoft/bitnet-embedding-0.6b | BitNet嵌入1.58-bit量化超轻量 | 0.6B | ~0.12 | 嵌入 |
| 98 | numind/NuExtract3 | 文档信息抽取结构化实体关系 | 5B | ~10 | OCR |
| 99 | black-forest-labs/FLUX.1-dev | FLUX.1开发版经典高质量文生图 | 12B | ~24 | 图像生成 |
| 100 | microsoft/Fara1.5-9B | 微软Fara 9B文档理解和OCR | 9B | ~18 | OCR |

### 3.4 显存分布

| 显存区间 | 模型数量 | 占比 |
|----------|----------|------|
| ≤ 2GB | 36 | 36% |
| 2~8GB | 28 | 28% |
| 8~16GB | 17 | 17% |
| 16~24GB | 10 | 10% |
| 24~32GB | 6 | 6% |
| 32~40GB | 3 | 3% |

> 64% 的模型显存需求 ≤ 8GB，可在单张 24GB 显卡上并行部署多个服务。

---

## 4. 基础镜像方案

### 4.1 镜像分层设计

| 层 | 内容 | 是否固化 | 体积估算 |
|----|------|----------|----------|
| **L1 系统层** | CUDA 12.1 + cuDNN + Python 3.11 + ffmpeg + 中文字体 + tesseract + 构建工具 | 固化 | ~4GB |
| **L2 框架层** | PyTorch 2.4 (cu121) + transformers + accelerate + huggingface_hub + safetensors + bitsandbytes + onnxruntime-gpu | 固化 | ~5GB |
| **L3 专用依赖** | paddleocr / diffusers / funasr / gguf 等按模型类别 | **不固化**，运行时装 | 动态 |
| **L4 业务文件** | requirements / 脚本 / 配置 / 业务代码 | **不固化**，宿主机挂载 | 0 |
| **L5 模型权重** | 模型权重、HF 缓存 | **不固化**，持久卷挂载 | 0 |

镜像总体积约 **9GB**，构建一次后稳定复用。

### 4.2 镜像内容定义

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

### 4.3 宿主机目录结构（所有动态内容）

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

### 4.4 环境变量设计

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

### 4.5 部署编排示例

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

### 4.6 entrypoint.sh 职责

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

### 4.7 冷启动优化

| 场景 | 首次启动 | 二次启动（缓存命中） |
|------|----------|---------------------|
| pip 装包 | 2~5 分钟 | < 10 秒 |
| 模型权重下载 | 取决于模型大小和带宽 | 0 秒（本地命中） |
| 服务启动 | 10~30 秒 | 10~30 秒 |

> 生产环境建议提前预热：对每个类别执行一次空跑，让 pip 缓存和权重缓存填充到持久卷。

---

## 5. CI/CD 持续集成方案

### 5.1 推荐项目结构

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

### 5.2 镜像构建流水线（仓库1）

触发条件：Dockerfile 或 requirements-base.txt 变更

```
触发 → Docker build → 推送到镜像仓库 → 打 Tag
        ↓
   安全扫描（可选）
        ↓
   更新 latest tag
```

CI 配置示例（GitHub Actions）：

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

### 5.3 部署配置流水线（仓库2）

触发条件：requirements / scripts / config / app 任意变更

```
代码推送 → 语法检查 → 同步到部署服务器 → 滚动重启受影响服务
```

CI 配置示例：

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

### 5.4 持续修改流程

| 修改场景 | 操作 | 是否重建镜像 | CI 触发 |
|----------|------|-------------|---------|
| 新增一个模型 | 改 config/models.yaml + docker-compose 加服务 | 否 | 配置仓库 CI |
| 修改某类 pip 依赖 | 改 requirements/xxx.txt | 否 | 配置仓库 CI |
| 升级 transformers 版本 | 改 requirements-base.txt | 是 | 镜像仓库 CI |
| 升级 PyTorch 大版本 | 改 Dockerfile + requirements-base.txt | 是 | 镜像仓库 CI |
| 修改启动逻辑 | 改 scripts/entrypoint.sh | 否 | 配置仓库 CI |
| 修改业务代码 | 改 app/server.py | 否 | 配置仓库 CI |

### 5.5 版本管理建议

- 镜像版本：`model-deploy:1.0` → `1.1`（小升级）→ `2.0`（大升级）
- 配置版本：Git 分支管理，main 为生产环境，dev 为测试环境
- 模型权重版本：通过 `MODEL_ID` 环境变量控制，HF 上游版本变更时更新

---

## 6. 注意事项与已知约束

### 6.1 模型兼容性约束

| 问题 | 影响 | 规避方式 |
|------|------|----------|
| PyTorch 版本 | 95% 模型可用 torch 2.4 (cu121)；极少数 2026 新模型可能需 torch 2.5+ | 镜像固化 torch 2.4 为主力；个别模型在 entrypoint 里 `pip install -U torch` 覆盖 |
| PaddlePaddle 自带 CUDA | paddlepaddle-gpu 捆绑 CUDA 库，与镜像 CUDA 共存基本无冲突 | 运行时装即可，无需特殊处理 |
| GGUF 模型（序号 67/68） | 走 llama-cpp-python 而非 PyTorch | 在 code-gen.txt 中加 llama-cpp-python |
| bitsandbytes 版本匹配 | 必须与 torch/CUDA 版本严格匹配 | 镜像固化匹配版本；若 entrypoint 升级 torch 则同步重装 bitsandbytes |
| flash-attn 编译慢 | 体积大、编译 10+ 分钟，仅 Ampere+ 显卡可用 | 默认不装，按需在 requirements 中解锁 |
| 触顶模型（序号 31/59） | ~40GB FP16 刚好触顶 40GB 约束 | 使用量化版本（INT8 降至 ~20GB）或独占 A100 40GB |

### 6.2 部署运维约束

| 问题 | 说明 |
|------|------|
| 冷启动延迟 | 首次装某类专用包 2~5 分钟；pip 缓存卷持久化后二次启动降至秒级 |
| 权重下载 | 首次拉取大模型（如 32B）可能耗时较长；建议离线预下载到持久卷 |
| 国内网络 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像站 |
| 多服务共卡 | 显存 ≤ 8GB 的模型可同卡并行；触顶模型独占一张卡 |
| 宿主机驱动 | 必须安装 NVIDIA 驱动 + NVIDIA Container Toolkit，否则容器无法访问 GPU |

### 6.3 显存估算参考

| 精度 | 估算公式 |
|------|----------|
| FP16 / BF16 | 参数量(B) × 2 ≈ 显存(GB) |
| INT8 / FP8 | 约为 FP16 的 50% |
| INT4 / NF4 / FP4 | 约为 FP16 的 25% |
| 1.58-bit (BitNet) | 约为 FP16 的 10% |

> 实际部署还需考虑框架开销、KV Cache 等，以上为模型权重估算值。

---

## 附录：手动补充模型

以下 4 个模型为手动补充（不在 Trending 筛选范围内），已计入总数：

| 序号 | 模型 | 类型 |
|------|------|------|
| 13 | deepseek-ai/DeepSeek-OCR | OCR |
| 14 | PaddlePaddle/PaddleOCR-VL-1.6 | OCR |
| 15 | PaddlePaddle/PP-OCRv5 | OCR |
| 16 | tencent/HunyuanOCR | OCR |
