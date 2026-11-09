"""按根配置的 logging 段设置进程日志输出。"""

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from hivememory.components.events.bus import RUNTIME_EVENT_RECORD_LOGGER
from hivememory.components.trace_context import TraceInjectFilter
from hivememory.config.app import LoggingConfig

# 本模块装到根 logger 上的输出；重新配置时只替换这些，测试框架与 WebSocket
# 广播等其他来源挂上的 handler 原样保留。
_installed_handlers: list[logging.Handler] = []


def configure_logging(config: LoggingConfig) -> None:
    """按 ``level``、``format``、``console_output``、``file_path`` 设置根 logger 的输出。

    控制台与文件输出只输出 ``level`` 及以上的记录。文件按本地日期切分：``file_path``
    始终是当天的日志，跨过零点后的第一条记录触发滚动，此前的内容改名为
    ``<file_path>.YYYY-MM-DD``（日期为这些内容所属的那一天），归档全部保留。开启
    WebSocket 广播且广播级别更低时，根 logger 的级别放宽到广播级别，低于 ``level`` 的
    记录只由广播 handler 按自己的级别过滤。两项输出都关闭时装上空输出，避免 Python 的
    兜底 handler 继续向 stderr 打印警告。

    uvicorn 的服务日志（启动、关闭与 ASGI 异常）与 ``warnings`` 的警告也交给根 logger，
    和应用日志同一格式、同一去处；每个请求已由请求日志中间件在 debug 级别记录，uvicorn 的
    访问日志关闭。MCP SDK 的 logger 只输出警告及以上。

    运行时事件的记录按 ``event_file_path`` 单独设置，见 :func:`_configure_event_record`。
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
        handlers.append(TimedRotatingFileHandler(path, when="midnight", encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())

    formatter = logging.Formatter(config.format)
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(TraceInjectFilter())
        root.addHandler(handler)
        _installed_handlers.append(handler)

    # uvicorn 在应用启动前已给自己的 logger 装上输出并阻止冒泡，这里撤掉它们。
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    logging.captureWarnings(True)
    # MCP SDK 以无状态方式处理每个请求，每次都在 INFO 级别记录会话结束；只保留它的警告与错误。
    logging.getLogger("mcp").setLevel(logging.WARNING)

    if config.websocket_enabled:
        level = min(level, logging.getLevelNamesMapping()[config.websocket_level])
    root.setLevel(level)

    _configure_event_record(config.event_file_path)


def _configure_event_record(file_path: str | None) -> None:
    """按 ``file_path`` 装上运行时事件记录的文件输出；为空时不装，事件不被记录。

    这个 logger 只由本函数装输出，重新配置时整体替换；它不向上传递，也不受 ``level``
    影响。文件与日志文件同样按本地日期切分，每行只有事件的 JSON。
    """
    record = logging.getLogger(RUNTIME_EVENT_RECORD_LOGGER)
    for handler in record.handlers[:]:
        record.removeHandler(handler)
        handler.close()
    if not file_path:
        return
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(path, when="midnight", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    record.addHandler(handler)
