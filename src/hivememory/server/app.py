"""HiveMemory FastAPI 应用入口"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from hivememory.server.deps import (
    init_system,
    get_system,
    init_websocket_log_broadcasting,
    shutdown_system,
    shutdown_websocket_log_broadcasting,
)
from hivememory.server.models.common import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 — 初始化/销毁 HiveMemorySystem 单例"""
    loop = asyncio.get_running_loop()

    def _loop_exception_handler(_loop, context):
        exc = context.get("exception")
        if exc is not None:
            logger.error(f"事件循环未处理异常: {exc}", exc_info=exc)
            return
        logger.error(f"事件循环未处理异常: {context.get('message', 'unknown')}")

    loop.set_exception_handler(_loop_exception_handler)
    logger.info("正在初始化 HiveMemorySystem...")
    system = init_system()
    await system.start()

    # 初始化 WebSocket 日志广播
    ws_manager = init_websocket_log_broadcasting(system.config)
    app.state.ws_manager = ws_manager  # 存储到 app state

    # 后台预热推理模型（不阻塞服务启动）
    asyncio.create_task(system.readiness_service.warmup_models())

    logger.info("HiveMemorySystem 就绪，服务启动完成")
    yield

    await shutdown_system()

    # 清理 WebSocket 连接
    await shutdown_websocket_log_broadcasting(ws_manager)
    logger.info("HiveMemorySystem 已关闭")


app = FastAPI(
    title="HiveMemory API",
    description="HiveMemory 记忆系统 HTTP API",
    version="0.1.0-beta",
    lifespan=lifespan,
)

# CORS 中间件 — 允许前端开发服务器跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:6918",      # Custom frontend port
        "http://127.0.0.1:6918",      # Custom frontend port
        "http://localhost:3000",      # Legacy port
        "http://127.0.0.1:3000",      # Legacy port
        "http://localhost:5173",      # Vite default (may be reserved by Windows)
        "http://127.0.0.1:5173",      # Vite default (may be reserved by Windows)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.0f}ms)")
    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# 健康检查 (Liveness)
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


# 就绪检查 (Readiness) — 模型是否已加载
@app.get("/health/ready")
async def readiness():
    system = get_system()
    readiness_state = await system.readiness_service.readiness()
    if readiness_state["models_ready"]:
        return ReadinessResponse(status="ready", models_ready=True)
    return JSONResponse(
        status_code=503,
        content=readiness_state,
    )


# 注册路由
from hivememory.server.routers.agents import router as agents_router
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.config import router as config_router
from hivememory.server.routers.ingest import router as ingest_router
from hivememory.server.routers.logs import router as logs_router
from hivememory.server.routers.memories import router as memories_router
from hivememory.server.routers.memory_tasks import router as memory_tasks_router
from hivememory.server.routers.runtime_events import router as runtime_events_router
from hivememory.server.routers.topics import router as topics_router

app.include_router(agents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(memory_tasks_router, prefix="/api/v1")
app.include_router(runtime_events_router, prefix="/api/v1")
app.include_router(topics_router, prefix="/api/v1")

# ==========================================
# 前后端整合与静态资源挂载 (生产环境与开发环境切换)
# ==========================================

# 默认情况下，前端由 Vite 独立启动（开发模式）。
# 若设置 HIVEMEMORY_SERVE_FRONTEND=true，则由 FastAPI 直接提供构建好的前端页面（生产/测试模式）。
SERVE_FRONTEND = os.getenv("HIVEMEMORY_SERVE_FRONTEND", "false").lower() == "true"

# 定位前端构建产物目录
# app.py 所在目录向上 4 级到达项目根目录，再进入 frontend/dist
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DEFAULT_FRONTEND_DIST_DIR = os.path.join(PROJECT_ROOT, "frontend", "dist")
FRONTEND_DIST_DIR = os.path.abspath(
    os.path.expanduser(os.getenv("HIVEMEMORY_FRONTEND_DIST_DIR", DEFAULT_FRONTEND_DIST_DIR))
)

if SERVE_FRONTEND:
    if os.path.isdir(FRONTEND_DIST_DIR):
        logger.info(f"启用前端静态资源代理: {FRONTEND_DIST_DIR}")
        
        # 挂载前端静态资源文件夹（Vite 构建通常在 assets 下）
        assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
            
        # 挂载其他可能存在的静态目录或文件（根据需要可进一步细化）
        # app.mount(...) 
        
        # SPA (单页应用) 兜底路由
        # 将所有未匹配的非 API 请求转发给 index.html，由前端 React Router 接管
        @app.get("/{catchall:path}")
        async def serve_spa(catchall: str):
            if catchall == "api" or catchall.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": "Not Found"})
                
            dist_root = os.path.realpath(FRONTEND_DIST_DIR)
            requested_path = os.path.realpath(os.path.join(dist_root, catchall))
            try:
                if os.path.commonpath([dist_root, requested_path]) != dist_root:
                    return JSONResponse(status_code=404, content={"error": "Not Found"})
            except ValueError:
                return JSONResponse(status_code=404, content={"error": "Not Found"})

            if os.path.isfile(requested_path):
                return FileResponse(requested_path)
                
            index_path = os.path.join(dist_root, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            else:
                return JSONResponse(status_code=404, content={"error": "Frontend index.html not found"})
    else:
        logger.warning(
            f"未找到前端构建目录 {FRONTEND_DIST_DIR}。请先在 frontend 目录下执行 'npm run build'。"
        )
