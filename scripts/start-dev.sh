#!/bin/bash
# HiveMemory 开发环境启动脚本

echo "=========================================="
echo "HiveMemory 开发环境启动"
echo "=========================================="
echo ""

# 检查是否在项目根目录
if [ ! -f "pyproject.toml" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 启动后端
echo "启动后端服务器..."
echo "后端地址: http://localhost:8769"
echo "API 文档: http://localhost:8769/docs"
echo ""

# 使用 uvicorn 启动后端（带自动重载）
uvicorn hivememory.server.app:app --host 0.0.0.0 --port 8769 --reload &
BACKEND_PID=$!

# 等待后端启动
echo "等待后端启动..."
sleep 3

# 检查后端是否启动成功
if curl -s http://localhost:8769/health > /dev/null; then
    echo "✓ 后端启动成功"
else
    echo "✗ 后端启动失败"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "启动前端开发服务器..."
echo "前端地址: http://localhost:5173"
echo ""

# 启动前端
cd frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "开发环境已启动"
echo "=========================================="
echo "后端: http://localhost:8769"
echo "前端: http://localhost:5173"
echo "API 文档: http://localhost:8769/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="

# 捕获 Ctrl+C 信号
trap "echo ''; echo '正在停止服务...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT

# 等待
wait
