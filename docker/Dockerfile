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
RUN [ "$PRELOAD_MODELS" = "true" ] || exit 0; \
    python - <<'PY'
import yaml
from FlagEmbedding import BGEM3FlagModel, FlagReranker

with open("/app/configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

embedding_cfg = (cfg.get("embedding") or {}).get("default") or {}
embedding_model = embedding_cfg.get("model_name", "BAAI/bge-m3")
embedding_device = embedding_cfg.get("device", "cpu")

reranker_cfg = (((cfg.get("retrieval") or {}).get("retriever") or {}).get("reranker") or {})
reranker_model = reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
reranker_device = reranker_cfg.get("device", "cpu")
reranker_use_fp16 = bool(reranker_cfg.get("use_fp16", True)) and reranker_device != "cpu"

print(f"[preload] embedding={embedding_model} device={embedding_device}")
BGEM3FlagModel(
    model_name_or_path=embedding_model,
    device=embedding_device,
    use_fp16=embedding_device != "cpu",
    show_progress_bar=False,
)

print(f"[preload] reranker={reranker_model} device={reranker_device}")
FlagReranker(
    model_name_or_path=reranker_model,
    device=reranker_device,
    use_fp16=reranker_use_fp16,
)
print("[preload] model predownload completed")
PY

# 5. 从阶段 1 复制构建好的前端静态文件
# 放置到容器中预期的位置
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 暴露端口
EXPOSE 8000

# 启动 FastAPI 应用 (使用 Uvicorn)
CMD ["uvicorn", "hivememory.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
