"""
WebSocket Connection Manager - 管理 WebSocket 连接和日志广播

职责：
- 跟踪活跃的 WebSocket 连接
- 广播消息到所有连接的客户端
- 自动清理断开的连接
- 缓冲日志（无客户端时）
- 向新客户端发送缓冲日志（catch-up）
"""

import asyncio
import logging
import uuid
from collections import deque
from typing import Any, Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    WebSocket 连接管理器 - 用于日志广播

    Features:
    - 多客户端连接跟踪
    - 广播消息到所有客户端
    - 自动清理死连接
    - 循环缓冲区（无客户端时）
    - 新客户端 catch-up（发送缓冲日志）
    """

    def __init__(self, buffer_size: int = 100):
        """
        初始化连接管理器

        Args:
            buffer_size: 缓冲区大小（无客户端时保留的日志数）
        """
        self._connections: Dict[str, WebSocket] = {}
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()
        logger.info(f"WebSocketConnectionManager initialized with buffer_size={buffer_size}")

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        注册新的 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
            client_id: 客户端唯一标识
        """
        async with self._lock:
            await websocket.accept()
            self._connections[client_id] = websocket
            logger.info(f"Client {client_id} connected (total: {len(self._connections)})")

    async def disconnect(self, client_id: str) -> None:
        """
        断开并移除 WebSocket 连接

        Args:
            client_id: 客户端唯一标识
        """
        async with self._lock:
            if client_id in self._connections:
                del self._connections[client_id]
                logger.info(f"Client {client_id} disconnected (remaining: {len(self._connections)})")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        广播消息到所有连接的客户端

        如果没有客户端连接，消息会被缓冲。
        自动清理发送失败的连接。

        Args:
            message: 要广播的消息（JSON-serializable dict）
        """
        async with self._lock:
            # 如果没有客户端，缓冲消息
            if not self._connections:
                self._buffer.append(message)
                return

            # 广播到所有客户端
            dead_clients = []
            for client_id, ws in self._connections.items():
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to client {client_id}: {e}")
                    dead_clients.append(client_id)

            # 清理死连接
            for client_id in dead_clients:
                del self._connections[client_id]
                logger.info(f"Removed dead client {client_id}")

    async def send_buffered_logs(self, client_id: str) -> None:
        """
        向新连接的客户端发送缓冲的日志（catch-up）

        Args:
            client_id: 客户端唯一标识
        """
        async with self._lock:
            if client_id not in self._connections:
                return

            ws = self._connections[client_id]
            sent_count = 0

            for log_msg in self._buffer:
                try:
                    await ws.send_json(log_msg)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send buffered log to {client_id}: {e}")
                    break

            if sent_count > 0:
                logger.info(f"Sent {sent_count} buffered logs to client {client_id}")

    async def disconnect_all(self) -> None:
        """断开所有客户端连接（shutdown 时调用）"""
        async with self._lock:
            client_ids = list(self._connections.keys())
            for client_id in client_ids:
                try:
                    ws = self._connections[client_id]
                    await ws.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {client_id}: {e}")

            self._connections.clear()
            logger.info("All WebSocket connections closed")

    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self._connections)

    def generate_client_id(self) -> str:
        """生成唯一的客户端 ID"""
        return str(uuid.uuid4())


__all__ = ["WebSocketConnectionManager"]
