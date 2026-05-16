"""依赖注入 — HiveMemorySystem 单例管理"""

import logging
from typing import Optional

from fastapi import Header

from hivememory.core.constants import DEFAULT_USER_ID
from hivememory.infrastructure.log_handler import WebSocketLogHandler
from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager
from hivememory.system.config import HiveMemoryConfig
from hivememory.patchouli.system import PatchouliSystem
from hivememory.system import HiveMemorySystem

logger = logging.getLogger(__name__)

_system: Optional[HiveMemorySystem] = None
_ws_manager: Optional[WebSocketConnectionManager] = None


def init_system(config: Optional[HiveMemoryConfig] = None) -> HiveMemorySystem:
    """lifespan startup 时调用，组装并返回 HiveMemorySystem"""
    global _system
    _system = HiveMemorySystem.build(config=config)
    logger.info("HiveMemorySystem 组装完成")
    return _system


async def shutdown_system() -> None:
    """lifespan shutdown 时调用"""
    global _system
    if _system:
        await _system.stop()
        logger.info("HiveMemorySystem 已关闭")
    _system = None


def get_system() -> HiveMemorySystem:
    """FastAPI Depends 注入 — 获取 HiveMemorySystem 单例"""
    if _system is None:
        raise RuntimeError("HiveMemorySystem 未初始化，服务未正确启动")
    return _system


def get_patchouli() -> PatchouliSystem:
    """FastAPI Depends 注入 — 获取 PatchouliSystem（兼容性访问器）"""
    return get_system().patchouli


def get_user_id(x_user_id: str = Header(default=DEFAULT_USER_ID)) -> str:
    """从请求头 x-user-id 提取用户 ID"""
    return x_user_id


def init_websocket_log_broadcasting(
    config: HiveMemoryConfig
) -> Optional[WebSocketConnectionManager]:
    """
    初始化 WebSocket 日志广播系统

    Steps:
    1. 检查配置是否启用
    2. 创建 WebSocketConnectionManager
    3. 创建 WebSocketLogHandler 并配置命名空间过滤
    4. 将 handler 附加到 root logger
    5. 配置日志级别和速率限制

    Args:
        config: HiveMemory 配置对象

    Returns:
        WebSocketConnectionManager 实例，如果未启用则返回 None
    """
    global _ws_manager

    if not config.logging.websocket_enabled:
        logger.info("WebSocket log broadcasting disabled")
        return None

    # 创建连接管理器
    _ws_manager = WebSocketConnectionManager(
        buffer_size=config.logging.websocket_buffer_size
    )

    # 创建日志处理器
    handler = WebSocketLogHandler(
        ws_manager=_ws_manager,
        namespaces=config.logging.websocket_namespaces,
        level=getattr(logging, config.logging.websocket_level),
        max_rate=config.logging.websocket_max_rate,
    )

    # 附加到 root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    # 注册追踪上下文过滤器
    from hivememory.infrastructure.trace_context import TraceInjectFilter
    handler.addFilter(TraceInjectFilter())

    logger.info(
        f"WebSocket log broadcasting initialized: "
        f"namespaces={config.logging.websocket_namespaces}, "
        f"level={config.logging.websocket_level}"
    )

    return _ws_manager


async def shutdown_websocket_log_broadcasting(
    manager: Optional[WebSocketConnectionManager]
) -> None:
    """
    关闭 WebSocket 日志广播系统

    清理所有客户端连接并从 root logger 移除 handler

    Args:
        manager: WebSocketConnectionManager 实例
    """
    if manager:
        # 断开所有客户端
        await manager.disconnect_all()

        # 从 root logger 移除 handler
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, WebSocketLogHandler):
                root_logger.removeHandler(handler)
                logger.info("WebSocket log handler removed")


def get_ws_manager() -> WebSocketConnectionManager:
    """FastAPI Depends 注入 — 获取 WebSocket 连接管理器"""
    if _ws_manager is None:
        raise RuntimeError("WebSocket manager not initialized")
    return _ws_manager
