"""入口按 logging 配置段设置根 logger 的输出。"""

import logging

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
def _restore_root_logger():
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


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
