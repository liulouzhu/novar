@echo off
chcp 65001 > nul
REM Novare Web 启动脚本
REM 同时启动 FastAPI 后端和 Vite 前端开发服务器

echo ========================================
echo   Novare Web - 启动中...
echo ========================================
echo.

REM 启动 FastAPI 后端 (端口 8000)
echo [1/2] 启动 FastAPI 后端 (port 8000)...
start "Novare Backend" cmd /c "cd /d %~dp0\.. && python -m uvicorn web.backend.app:app --host 0.0.0.0 --port 8000 --reload"

REM 等待后端启动
timeout /t 3 /nobreak > nul

REM 启动 Vite 前端 (端口 5173)
echo [2/2] 启动 Vite 前端 (port 5173)...
start "Novare Frontend" cmd /c "cd /d %~dp0\frontend && npm run dev"

echo.
echo ========================================
echo   启动完成！
echo   前端: http://localhost:5173
echo   后端: http://localhost:8000
echo   API 文档: http://localhost:8000/docs
echo ========================================
echo.
pause
