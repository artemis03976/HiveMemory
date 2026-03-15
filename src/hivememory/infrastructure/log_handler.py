"""
WebSocket Log Handler - 自定义日志处理器，将日志广播到 WebSocket 客户端

职责：
- 拦截特定命名空间的日志
- 将 LogRecord 转换为 JSON 格式
- 通过 WebSocket 广播日志
- 非阻塞异步调度
- 速率限制防止淹没客户端
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Set

from hivememory.infrastructure.rate_limiter import RateLimiter
from hivememory.infrastructure.websocket_manager import WebSocketConnectionManager

# LogRecord 的标准属性（用于过滤 extra 字段）
STANDARD_RECORD_ATTRS = {
    "name", "msg", "args", "created", "filename", "funcName", "levelname",
    "levelno", "lineno", "module", "msecs", "message", "pathname", "process",
    "processName", "relativeCreated", "thread", "threadName", "exc_info",
    "exc_text", "stack_info", "getMessage", "asctime",
}

# 消息大小限制
MAX_MESSAGE_LENGTH = 10000  # 10KB
MAX_TRACEBACK_LENGTH = 5000  # 5KB


class WebSocketLogHandler(logging.Handler):
    """
    自定义日志处理器 - 将日志广播到 WebSocket 客户端

    Features:
    - 命名空间过滤（支持通配符 '*'）
    - JSON 格式化
    - 异步非阻塞广播
    - 速率限制
    - 消息大小限制
    """

    def __init__(
        self,
        ws_manager: WebSocketConnectionManager,
        namespaces: List[str],
        level: int = logging.INFO,
        max_rate: int = 100,
    ):
        """
        初始化 WebSocket 日志处理器

        Args:
            ws_manager: WebSocket 连接管理器
            namespaces: 要拦截的命名空间列表（支持 '.*' 通配符）
            level: 最低日志级别
            max_rate: 最大广播速率（条/秒）
        """
        super().__init__(level)
        self.ws_manager = ws_manager
        self.namespaces = namespaces
        self._rate_limiter = RateLimiter(max_rate)
        self._logger = logging.getLogger(__name__)

    def emit(self, record: logging.LogRecord) -> None:
        """
        处理日志记录（由 logging 系统调用）

        Flow:
        1. 检查命名空间过滤
        2. 检查速率限制
        3. 格式化为 JSON
        4. 异步调度广播（非阻塞）
        """
        try:
            # 命名空间过滤
            if not self._should_handle(record):
                return

            # 速率限制（ERROR 及以上总是发送）
            if record.levelno < logging.ERROR and not self._rate_limiter.allow():
                return

            # 格式化为 JSON
            log_data = self._format_log_record(record)

            # 异步调度广播（非阻塞）
            self._schedule_broadcast(log_data)

        except Exception:
            # 永远不要让日志处理器崩溃
            self.handleError(record)

    def _should_handle(self, record: logging.LogRecord) -> bool:
        """
        检查日志记录是否匹配配置的命名空间

        支持：
        - 通配符: 'hivememory.*' 匹配 'hivememory.patchouli.kernel'
        - 精确匹配: 'hivememory.patchouli.kernel' 只匹配该 logger

        Args:
            record: 日志记录

        Returns:
            True if should handle, False otherwise
        """
        logger_name = record.name

        for namespace in self.namespaces:
            if namespace.endswith(".*"):
                # 通配符匹配
                prefix = namespace[:-2]
                if logger_name.startswith(prefix):
                    return True
            elif logger_name == namespace:
                # 精确匹配
                return True

        return False

    def _format_log_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        将 LogRecord 转换为 JSON-serializable dict

        Args:
            record: 日志记录

        Returns:
            JSON-serializable dict
        """
        # 基础字段
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process,
        }

        # 异常信息
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_value) if exc_value else "",
                "traceback": self.formatter.formatException(record.exc_info) if self.formatter else "",
            }

        # 自定义字段（extra）
        extra = {}
        for key, value in record.__dict__.items():
            if key not in STANDARD_RECORD_ATTRS:
                try:
                    # 只包含 JSON-serializable 的值
                    json.dumps(value)
                    extra[key] = value
                except (TypeError, ValueError):
                    pass

        if extra:
            log_data["extra"] = extra

        # 消息大小限制
        log_data = self._truncate_if_needed(log_data)

        return log_data

    def _truncate_if_needed(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        截断过大的字段，防止内存问题

        Args:
            log_data: 日志数据

        Returns:
            截断后的日志数据
        """
        # 截断消息
        if len(log_data["message"]) > MAX_MESSAGE_LENGTH:
            log_data["message"] = (
                log_data["message"][:MAX_MESSAGE_LENGTH] + "... [truncated]"
            )

        # 截断 traceback
        if "exception" in log_data:
            tb = log_data["exception"]["traceback"]
            if len(tb) > MAX_TRACEBACK_LENGTH:
                log_data["exception"]["traceback"] = (
                    tb[:MAX_TRACEBACK_LENGTH] + "\n... [truncated]"
                )

        return log_data

    def _schedule_broadcast(self, log_data: Dict[str, Any]) -> None:
        """
        异步调度广播（非阻塞）

        使用 asyncio.create_task() 在事件循环中调度广播，
        不阻塞日志系统。

        Args:
            log_data: 要广播的日志数据
        """
        try:
            # 获取运行中的事件循环
            loop = asyncio.get_running_loop()
            # 调度异步广播任务
            loop.create_task(self.ws_manager.broadcast(log_data))
        except RuntimeError:
            # 没有运行中的事件循环（不应该发生在 FastAPI 中）
            # 静默跳过，不影响日志系统
            pass


__all__ = ["WebSocketLogHandler"]
