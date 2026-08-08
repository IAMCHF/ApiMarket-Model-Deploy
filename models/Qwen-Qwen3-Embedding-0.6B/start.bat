@echo off
rem ============================================================================
rem 统一启动脚本 — 固定监听 0.0.0.0:8080
rem
rem 用法：双击 start.bat 或在命令行执行
rem 权重：请提前将模型权重放置于本目录 weights/ 下（见 weights/README.txt）
rem ============================================================================
cd /d "%~dp0"

echo [start] 安装依赖...
python -m pip install -r requirements.txt

echo [start] 启动服务 http://0.0.0.0:8080 ...
python server.py
