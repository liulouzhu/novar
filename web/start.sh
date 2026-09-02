#!/bin/bash
# Novare Web 启动脚本

echo "========================================"
echo "  Novare Web - 启动中..."
echo "========================================"
echo ""

# 启动 FastAPI 后端
echo "[1/2] 启动 FastAPI 后端 (port 8000)..."
cd "$(dirname "$0")/.." && python -m uvicorn web.backend.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 3

# 启动 Vite 前端
echo "[2/2] 启动 Vite 前端 (port 5173)..."
cd "$(dirname "$0")/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "  启动完成！"
echo "  前端: http://localhost:5173"
echo "  后端: http://localhost:8000"
echo "  API 文档: http://localhost:8000/docs"
echo "========================================"

# 等待任一进程退出
wait $BACKEND_PID $FRONTEND_PID
