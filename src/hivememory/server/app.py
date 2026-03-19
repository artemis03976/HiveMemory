"""HiveMemory FastAPI 应用入口"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    """应用生命周期管理 — 初始化/销毁 PatchouliSystem 单例"""
    loop = asyncio.get_running_loop()

    def _loop_exception_handler(_loop, context):
        exc = context.get("exception")
        if exc is not None:
            logger.error(f"事件循环未处理异常: {exc}", exc_info=exc)
            return
        logger.error(f"事件循环未处理异常: {context.get('message', 'unknown')}")

    loop.set_exception_handler(_loop_exception_handler)
    logger.info("正在初始化 PatchouliSystem...")
    system = init_system()
    system.start_observer_idle_monitor(lazy_start=True)

    # 初始化 WebSocket 日志广播
    ws_manager = init_websocket_log_broadcasting(system.config)
    app.state.ws_manager = ws_manager  # 存储到 app state

    # 后台预热推理模型（不阻塞服务启动）
    asyncio.create_task(system.kernel.warmup_models())

    logger.info("PatchouliSystem 就绪，服务启动完成")
    yield

    # 清理 WebSocket 连接
    await shutdown_websocket_log_broadcasting(ws_manager)

    shutdown_system()
    logger.info("PatchouliSystem 已关闭")


app = FastAPI(
    title="HiveMemory API",
    description="HiveMemory 记忆系统 HTTP API",
    version="0.1.0",
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
    ready = system.kernel.is_models_ready()
    if ready:
        return ReadinessResponse(status="ready", models_ready=True)
    return JSONResponse(
        status_code=503,
        content={"status": "warming_up", "models_ready": False},
    )


# 注册路由
from hivememory.server.routers.chat import router as chat_router
from hivememory.server.routers.config import router as config_router
from hivememory.server.routers.ingest import router as ingest_router
from hivememory.server.routers.logs import router as logs_router
from hivememory.server.routers.memories import router as memories_router
from hivememory.server.routers.topics import router as topics_router

app.include_router(chat_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(topics_router, prefix="/api/v1")
