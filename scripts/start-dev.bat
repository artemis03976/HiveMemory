@echo off
chcp 65001 >nul
REM HiveMemory 开发环境启动脚本 (Windows) - 修复版

echo ==========================================
echo HiveMemory 开发环境启动
echo ==========================================
echo.

REM 检查是否在项目根目录
if not exist "pyproject.toml" (
    echo [错误] 请在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 检查 conda 环境
echo 检查 Python 环境...
python -c "import sys; print(f'当前环境: {sys.prefix}')"

REM 检查依赖
echo 检查依赖包...
python -c "import sse_starlette" 2>nul
if %errorlevel% neq 0 (
    echo [警告] 缺少 sse-starlette 包
    echo 正在安装依赖...
    pip install sse-starlette
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败
        echo.
        echo 请手动安装依赖:
        echo   pip install sse-starlette
        echo.
        echo 或激活 hivememory 环境:
        echo   conda activate hivememory
        pause
        exit /b 1
    )
)

echo [OK] 依赖检查通过
echo.

REM 测试后端导入
echo 测试后端模块...
python -c "from hivememory.server.app import app; print('[OK] 后端模块正常')" 2>nul
if %errorlevel% neq 0 (
    echo [错误] 后端模块导入失败
    echo.
    echo 请检查:
    echo   1. 是否激活了正确的 conda 环境
    echo   2. 是否安装了所有依赖: pip install -e .
    echo.
    python -c "from hivememory.server.app import app"
    pause
    exit /b 1
)

echo.
echo 启动后端服务器...
echo 后端地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.

REM 启动后端（在新窗口）
start "HiveMemory Backend" cmd /k "title HiveMemory Backend && python -m hivememory.server"

REM 等待后端启动
echo 等待后端启动...
timeout /t 5 /nobreak > nul

REM 检查后端是否启动成功
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] 后端启动成功
) else (
    echo [警告] 后端可能未完全启动，请检查后端窗口
    echo 继续启动前端...
)

echo.
echo 启动前端开发服务器...
echo 前端地址: http://localhost:5173
echo.

REM 检查前端目录
if not exist "frontend\package.json" (
    echo [错误] 找不到前端目录
    pause
    exit /b 1
)

REM 检查 node_modules
if not exist "frontend\node_modules" (
    echo [警告] 前端依赖未安装
    echo 正在安装前端依赖...
    cd frontend
    call npm install
    if %errorlevel% neq 0 (
        echo [错误] 前端依赖安装失败
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo [OK] 前端依赖安装完成
)

REM 启动前端（在新窗口）
cd frontend
start "HiveMemory Frontend" cmd /k "title HiveMemory Frontend && npm run dev"
cd ..

echo.
echo ==========================================
echo 开发环境已启动
echo ==========================================
echo 后端: http://localhost:8000
echo 前端: http://localhost:5173
echo API 文档: http://localhost:8000/docs
echo.
echo 提示:
echo   - 后端和前端在独立窗口中运行
echo   - 关闭对应窗口可停止服务
echo   - 如遇到问题，请查看各窗口的日志
echo ==========================================
echo.

pause
