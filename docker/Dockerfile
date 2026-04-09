# ==========================================
# 阶段 1：构建前端 (Node.js)
# ==========================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# 1. 复制依赖描述文件并安装依赖
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# 2. 复制前端源代码并构建
COPY frontend/ ./
RUN npm run build


# ==========================================
# 阶段 2：构建后端及运行环境 (Python)
# ==========================================
FROM python:3.12-slim AS backend

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HIVEMEMORY_SERVE_FRONTEND=true \
    HIVEMEMORY_FRONTEND_DIST_DIR=/app/frontend/dist

WORKDIR /app

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. 复制后端代码和配置
COPY src/ ./src/
COPY configs/ ./configs/

# 4. 从阶段 1 复制构建好的前端静态文件
# 放置到容器中预期的位置
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 应用 (使用 Uvicorn)
CMD ["uvicorn", "hivememory.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
