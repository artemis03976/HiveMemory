"""
Trace Context - 分布式追踪上下文管理

使用 contextvars 实现协程安全的追踪上下文注入，支持：
- Trace ID: 完整业务流的唯一标识
- Span Name: 子任务/组件的名称
- Task Type: 前台任务 (foreground) 或后台任务 (background)

通过 TraceInjectFilter 自动将上下文注入到所有日志记录中，
无需修改现有的 logger.info() 调用。
"""

import contextvars
import logging
import uuid
from typing import Literal, Tuple

# 定义上下文变量
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default="system"
)
current_span_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_span_name", default="main"
)
current_task_type: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_task_type", default="foreground"
)


class TraceInjectFilter(logging.Filter):
    """
    日志过滤器 - 自动将追踪上下文注入到每条日志记录中

    在 LogRecord 对象上添加 trace_id, span_name, task_type 属性，
    供后续的 Handler 和 Formatter 使用。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace_id.get()
        record.span_name = current_span_name.get()
        record.task_type = current_task_type.get()
        return True


def generate_trace_id(prefix: str = "") -> str:
    """
    生成唯一的 Trace ID

    Args:
        prefix: 可选的前缀（如 "chat", "archive"）

    Returns:
        格式为 "{prefix}-{uuid8}" 或 "{uuid8}" 的字符串
    """
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}-{suffix}" if prefix else suffix


def set_trace_context(
    trace_id: str,
    span_name: str,
    task_type: Literal["foreground", "background"]
) -> Tuple[contextvars.Token, contextvars.Token, contextvars.Token]:
    """
    设置追踪上下文

    Args:
        trace_id: 追踪 ID
        span_name: Span 名称（如 "PatchouliSystem.Chat"）
        task_type: 任务类型（"foreground" 或 "background"）

    Returns:
        三个 Token 的元组，用于后续恢复上下文
    """
    return (
        current_trace_id.set(trace_id),
        current_span_name.set(span_name),
        current_task_type.set(task_type),
    )


def reset_trace_context(
    tokens: Tuple[contextvars.Token, contextvars.Token, contextvars.Token]
) -> None:
    """
    恢复追踪上下文到之前的状态

    Args:
        tokens: set_trace_context() 返回的 Token 元组
    """
    current_trace_id.reset(tokens[0])
    current_span_name.reset(tokens[1])
    current_task_type.reset(tokens[2])
