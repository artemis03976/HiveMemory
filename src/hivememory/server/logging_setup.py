"""按根配置的 logging 段设置进程日志输出。"""

import logging
from pathlib import Path

from hivememory.components.trace_context import TraceInjectFilter
from hivememory.config.app import LoggingConfig

# 本模块装到根 logger 上的输出；重新配置时只替换这些，测试框架与 WebSocket
# 广播等其他来源挂上的 handler 原样保留。
_installed_handlers: list[logging.Handler] = []


def configure_logging(config: LoggingConfig) -> None:
    """按 ``level``、``format``、``console_output``、``file_path`` 设置根 logger 的输出。

    控制台与文件输出只输出 ``level`` 及以上的记录。开启 WebSocket 广播且广播级别更低时，
    根 logger 的级别放宽到广播级别，低于 ``level`` 的记录只由广播 handler 按自己的
    级别过滤。两项输出都关闭时装上空输出，避免 Python 的兜底 handler 继续向 stderr
    打印警告。uvicorn 自己的访问与错误日志不经根 logger，不受本配置影响。
    """
    root = logging.getLogger()
    for handler in _installed_handlers:
        root.removeHandler(handler)
        handler.close()
    _installed_handlers.clear()

    level = logging.getLevelNamesMapping()[config.level]
    handlers: list[logging.Handler] = []
    if config.console_output:
        handlers.append(logging.StreamHandler())
    if config.file_path:
        path = Path(config.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())

    formatter = logging.Formatter(config.format)
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(TraceInjectFilter())
        root.addHandler(handler)
        _installed_handlers.append(handler)

    if config.websocket_enabled:
        level = min(level, logging.getLevelNamesMapping()[config.websocket_level])
    root.setLevel(level)
