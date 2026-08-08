模型权重放置说明
================

请将本模型的权重文件直接放置于本目录（weights/）下，目录结构与
Hugging Face 仓库保持一致。示例（transformers 类模型）：

    weights/
    ├── config.json
    ├── model.safetensors          # 或 model.bin / pytorch_model.bin
    ├── tokenizer.json
    └── ...                        # 其余仓库文件

启动服务（start.sh / start.bat）时，程序会优先从本目录加载权重；
若本目录为空，则回退为从 Hugging Face 在线加载。

特殊模型放置约定：
- GGUF 类模型（如 Qwen/Qwen2.5-Coder-7B-Instruct-GGUF）：
    将 .gguf 权重文件直接放在本目录。
- Grounding-DINO 类模型：
    将 config（*.yaml/config.py）与权重（*.pth/*.safetensors）放在本目录。
- 其余厂商专用格式（NeMo / CosyVoice / TRELLIS 等）：
    按对应官方仓库的 checkpoint 目录结构放置。
