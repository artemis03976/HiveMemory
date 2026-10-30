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
    HIVEMEMORY_FRONTEND_DIST_DIR=/app/frontend/dist \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

ARG PIP_PRIMARY_INDEX_URL=https://pypi.org/simple
ARG PIP_FALLBACK_INDEX_URL_1=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_FALLBACK_INDEX_URL_2=https://mirrors.aliyun.com/pypi/simple/
ARG PRELOAD_MODELS=true

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt ./
RUN set -eux; \
    for index_url in "$PIP_PRIMARY_INDEX_URL" "$PIP_FALLBACK_INDEX_URL_1" "$PIP_FALLBACK_INDEX_URL_2"; do \
        for attempt in 1 2 3; do \
            pip install --no-cache-dir --retries 10 --timeout 120 -i "$index_url" -r requirements.txt && exit 0; \
            if [ "$attempt" -eq 3 ]; then break; fi; \
            sleep $((attempt * 10)); \
        done; \
    done; \
    exit 1

# 3. 复制后端代码和配置
COPY src/ ./src/
COPY configs/ ./configs/

# 4. 预下载 embedding/reranker 模型（减少首次启动等待）
# 经运行时同一套配置加载与模型服务完成：模型名、自定义模型注册与缓存目录都与运行时一致。
RUN [ "$PRELOAD_MODELS" = "true" ] || exit 0; \
    python - <<'PY'
import asyncio

from hivememory.config.app import load_app_config
from hivememory.infrastructure.embedding import get_bge_m3_service
from hivememory.infrastructure.rerank import get_fast_embed_reranker_service

config = load_app_config("/app/configs/config.yaml")

embedding = get_bge_m3_service(config.shared.embedding.default)
print(f"[preload] embedding={embedding.model_name} cache_dir={embedding.cache_dir}")
asyncio.run(embedding.load())

reranker_config = config.patchouli.retrieval.reranker
if reranker_config.enabled:
    reranker = get_fast_embed_reranker_service(reranker_config)
    print(f"[preload] reranker={reranker.model_name} cache_dir={reranker.cache_dir}")
    asyncio.run(reranker.load())

print("[preload] model predownload completed")
PY

# 5. 从阶段 1 复制构建好的前端静态文件
# 放置到容器中预期的位置
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 应用 (使用 Uvicorn)
CMD ["uvicorn", "hivememory.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
