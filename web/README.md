# Novare Web 启动指南

## 环境要求

- Python >= 3.10
- Node.js >= 18
- PostgreSQL（必须配置 DATABASE_URL 环境变量）

## 快速启动

### 1. 安装后端依赖

```bash
cd d:\project\research-agent
pip install fastapi uvicorn[standard] websockets python-multipart
```

### 2. 安装前端依赖

```bash
cd d:\project\research-agent\web\frontend
npm install
```

### 3. 启动后端（FastAPI）

```bash
cd d:\project\research-agent
python -m uvicorn web.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

启动后可访问 API 文档：http://localhost:8000/docs

### 4. 启动前端（Vite）

新开一个终端：

```bash
cd d:\project\research-agent\web\frontend
npm run dev
```

### 5. 访问

浏览器打开 **http://localhost:5173**

---

## 一键启动（Windows）

```bash
cd d:\project\research-agent\web
start.bat
```

## 一键启动（Linux/Mac）

```bash
cd d:\project\research-agent/web
bash start.sh
```

---

## 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 (Vite dev server) | 5173 | 用户访问入口 |
| 后端 (FastAPI) | 8000 | API 服务，前端通过 Vite proxy 转发 |

前端的 `/api/*` 和 `/ws/*` 请求会自动代理到后端 8000 端口（由 `vite.config.ts` 配置）。

---

## 常见问题

### 后端启动报 `ModuleNotFoundError`

确保在项目根目录 `d:\project\research-agent` 下启动，且 `web.backend` 模块可被正确导入。

### 前端白屏 / 请求 404

确认后端已启动，且端口 8000 正常监听。可访问 http://localhost:8000/api/health 检查。

### WebSocket 连接失败

确认前后端端口配置一致，浏览器控制台查看 WS 连接地址是否为 `ws://localhost:5173/ws/chat/{session_id}`。

---

## Redis（可选依赖）

Redis 用于以下功能，**不启用时系统照常运行**（降级模式）：

| 功能 | Redis 可用时 | Redis 不可用时 |
|------|-------------|---------------|
| 会话并发锁 | 同一用户同一会话不会重入 | 无锁，可能重入 |
| 渠道消息去重 | 自动跳过重复投递 | 无去重，可能重复处理 |
| 任务状态查询 | `GET /api/chat/{id}/task` 返回实时状态 | 返回 `{"status":"idle"}` |
| 取消任务 | `POST /api/chat/{id}/cancel` 协作式取消 | 503 或降级为强杀 |

### 启用 Redis

```bash
# .env 中添加
NOVARE_REDIS_ENABLED=true
NOVARE_REDIS_URL=redis://localhost:6379/0
```

本地开发可用 Docker 快速启动 Redis：
```bash
docker run -d --name novare-redis -p 6379:6379 redis:7-alpine redis-server --appendonly yes
```
