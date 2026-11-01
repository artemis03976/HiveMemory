"""入口按 logging 配置段设置根 logger 的输出。"""

import logging
import os
import re
import time

import pytest

from hivememory.config.app import LoggingConfig
from hivememory.server.logging_setup import configure_logging

_FORMAT = "%(levelname)s|%(name)s|%(message)s"


class _RecordingHandler(logging.Handler):
    """代表测试框架或 WebSocket 广播挂在根 logger 上的其他 handler。"""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture(autouse=True)
def _restore_global_logging():
    """还原入口改动的全局状态：根 logger、uvicorn 的 logger 与 warnings 捕获。"""
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    uvicorn_state = [
        (logger, logger.handlers[:], logger.propagate, logger.disabled)
        for logger in (logging.getLogger("uvicorn"), logging.getLogger("uvicorn.access"))
    ]
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    for logger, logger_handlers, propagate, disabled in uvicorn_state:
        logger.handlers[:] = logger_handlers
        logger.propagate = propagate
        logger.disabled = disabled
    logging.captureWarnings(False)


@pytest.fixture
def log():
    return logging.getLogger("tests.logging_setup")


def test_console_outputs_records_at_or_above_level_in_configured_format(capsys, log):
    configure_logging(LoggingConfig(level="WARNING", format=_FORMAT))

    log.info("dropped")
    log.warning("kept")

    stderr = capsys.readouterr().err
    assert "WARNING|tests.logging_setup|kept" in stderr
    assert "dropped" not in stderr


def test_file_output_creates_directory_while_console_is_disabled(tmp_path, capsys, log):
    log_file = tmp_path / "logs" / "hivememory.log"
    configure_logging(
        LoggingConfig(level="INFO", format=_FORMAT, console_output=False, file_path=str(log_file))
    )

    log.info("to file")

    assert log_file.read_text(encoding="utf-8") == "INFO|tests.logging_setup|to file\n"
    assert "to file" not in capsys.readouterr().err


def test_file_output_moves_an_earlier_day_to_its_dated_name(tmp_path, log):
    log_file = tmp_path / "backend.log"
    log_file.write_text("earlier day\n", encoding="utf-8")
    earlier = time.time() - 2 * 24 * 3600
    os.utime(log_file, (earlier, earlier))
    configure_logging(LoggingConfig(console_output=False, file_path=str(log_file), format=_FORMAT))

    log.warning("today")

    (archived,) = [path for path in tmp_path.iterdir() if path != log_file]
    assert re.fullmatch(r"backend\.log\.\d{4}-\d{2}-\d{2}", archived.name)
    assert archived.read_text(encoding="utf-8") == "earlier day\n"
    assert log_file.read_text(encoding="utf-8") == "WARNING|tests.logging_setup|today\n"


def test_uvicorn_server_logs_join_outputs_and_access_logs_are_dropped(tmp_path):
    # uvicorn 启动时给自己的 logger 装上输出并阻止冒泡到根 logger。
    uvicorn_output = _RecordingHandler()
    for name in ("uvicorn", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.addHandler(uvicorn_output)
        uvicorn_logger.propagate = False
    log_file = tmp_path / "backend.log"
    configure_logging(LoggingConfig(console_output=False, file_path=str(log_file), format=_FORMAT))

    logging.getLogger("uvicorn.error").error("Exception in ASGI application")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /health HTTP/1.1" 200')

    assert log_file.read_text(encoding="utf-8") == (
        "ERROR|uvicorn.error|Exception in ASGI application\n"
    )
    assert uvicorn_output.messages == []


def test_reconfigure_replaces_own_outputs_and_keeps_other_handlers(tmp_path, log):
    other = _RecordingHandler()
    logging.getLogger().addHandler(other)
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    configure_logging(LoggingConfig(console_output=False, file_path=str(first)))

    configure_logging(LoggingConfig(console_output=False, file_path=str(second), format=_FORMAT))
    log.warning("after reconfigure")

    assert first.read_text(encoding="utf-8") == ""
    assert second.read_text(encoding="utf-8") == "WARNING|tests.logging_setup|after reconfigure\n"
    assert other.messages == ["after reconfigure"]


def test_lower_websocket_level_reaches_broadcast_without_loosening_outputs(capsys, log):
    broadcast = _RecordingHandler(logging.DEBUG)
    logging.getLogger().addHandler(broadcast)
    configure_logging(
        LoggingConfig(
            level="WARNING", format=_FORMAT, websocket_enabled=True, websocket_level="DEBUG"
        )
    )

    log.debug("detail")

    assert broadcast.messages == ["detail"]
    assert "detail" not in capsys.readouterr().err
