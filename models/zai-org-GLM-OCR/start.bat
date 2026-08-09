@echo off
rem ============================================================================
rem 统一启动脚本 — 固定监听 0.0.0.0:8080
rem
rem 用法：双击 start.bat 或在命令行执行
rem 第一步：换源（国内 pip 加速，不改镜像构建源；可用 PIP_INDEX_URL 环境变量覆盖）
rem 权重：请提前将模型权重放置于本目录 weights/ 下（见 weights/README.txt）
rem ============================================================================
cd /d "%~dp0"

rem ---- 1. 换源（国内 pip 加速；内网环境可设置 PIP_INDEX_URL 指向内网镜像） ----
if "%PIP_INDEX_URL%"=="" set PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
if "%PIP_TRUSTED_HOST%"=="" set PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
set PIP_DISABLE_PIP_VERSION_CHECK=1

rem ---- 2. 安装依赖（命中 /data/pip-cache 卷则秒级完成） ----
echo [start] 安装依赖（源: %PIP_INDEX_URL%）...
python -m pip install -r requirements.txt

rem ---- 3. 启动服务（固定 0.0.0.0:8080） ----
echo [start] 启动服务 http://0.0.0.0:8080 ...
python server.py
