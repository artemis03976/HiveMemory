"""
WebSocket Logs Router - 实时日志流式传输端点

提供 WebSocket 端点用于向前端广播后端日志
"""

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager
from hivememory.server.deps import get_ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["logs"])


@router.websocket("/ws/logs")
async def websocket_logs_endpoint(
    websocket: WebSocket,
    manager: WebSocketConnectionManager = Depends(get_ws_manager),
):
    """
    WebSocket 端点 - 实时日志流式传输

    Protocol:
    1. 客户端连接，服务器接受并生成 client_id
    2. 服务器发送缓冲的日志作为 catch-up
    3. 服务器实时广播新日志
    4. 客户端可发送 ping 保持连接
    5. 优雅处理断开连接

    Message Format (JSON):
    {
        "timestamp": "2026-03-15T10:30:45.123456",
        "level": "INFO",
        "logger": "hivememory.patchouli.runtime.core",
        "message": "Processing user request",
        "module": "kernel.py",
        "function": "handle_hot",
        "line": 142,
        "thread": "MainThread",
        "process": 12345,
        "exception": {...},  // Optional
        "extra": {...}       // Optional
    }
    """
    # 生成唯一客户端 ID
    client_id = manager.generate_client_id()

    try:
        # 接受连接并注册
        await manager.connect(websocket, client_id)

        # 发送缓冲的日志（catch-up）
        await manager.send_buffered_logs(client_id)

        # 保持连接，监听客户端消息（ping/pong）
        while True:
            try:
                # 接收客户端消息（用于 keepalive）
                data = await websocket.receive_text()
                # 可以处理 ping/pong 或其他控制消息
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Error in WebSocket connection {client_id}: {e}", exc_info=True)
    finally:
        # 清理连接
        await manager.disconnect(client_id)


__all__ = ["router"]
