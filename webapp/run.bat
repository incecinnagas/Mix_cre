@echo off
REM Windows批处理脚本启动Web应用
cd /d %~dp0
cd ..

echo ========================================
echo 启动 Transformer+SSM Web API
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查模型文件
if exist "results\best.ckpt" (
    set MODEL_CKPT=%CD%\results\best.ckpt
    echo [信息] 找到模型: %MODEL_CKPT%
) else if exist "results\checkpoints\best.ckpt" (
    set MODEL_CKPT=%CD%\results\checkpoints\best.ckpt
    echo [信息] 找到模型: %MODEL_CKPT%
) else (
    echo [警告] 未找到模型检查点
    echo        请设置 MODEL_CKPT 环境变量指向模型文件
    echo        或确保模型文件在 results\best.ckpt
)

REM 设置序列长度
if not defined SEQ_LEN set SEQ_LEN=3000

echo.
echo [信息] 工作目录: %CD%
echo [信息] 序列长度: %SEQ_LEN%
echo.
echo ========================================
echo 正在启动Web服务...
echo.
echo ⚠️  重要: 在浏览器中访问以下地址:
echo    http://localhost:8000
echo    或
echo    http://127.0.0.1:8000
echo.
echo (不要使用 0.0.0.0，那是服务器绑定地址)
echo.
echo 按 Ctrl+C 停止服务
echo ========================================
echo.

REM 启动服务
python webapp\run.py

if errorlevel 1 (
    echo.
    echo [错误] 启动失败
    echo 请检查:
    echo 1. 是否已安装依赖: pip install fastapi uvicorn pydantic numpy
    echo 2. 模型文件是否存在
    echo 3. 端口8000是否被占用
    pause
)


