import asyncio
import logging
import sys

import pytest

from hivememory.infrastructure.log_handler import (
    MAX_MESSAGE_LENGTH,
    MAX_TRACEBACK_LENGTH,
    WebSocketLogHandler,
)


class FakeManager:
    def __init__(self) -> None:
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


def _record(name: str = "hivememory.patchouli", level: int = logging.INFO, msg: str = "hello"):
    return logging.LogRecord(name, level, __file__, 12, msg, (), None)


def test_should_handle_exact_and_wildcard_namespaces():
    handler = WebSocketLogHandler(FakeManager(), ["hivememory.patchouli", "hivememory.alice.*"])

    assert handler._should_handle(_record("hivememory.patchouli")) is True
    assert handler._should_handle(_record("hivememory.alice.runtime")) is True
    assert handler._should_handle(_record("other")) is False


def test_format_log_record_includes_trace_context_and_json_extra():
    handler = WebSocketLogHandler(FakeManager(), ["hivememory.*"])
    record = _record()
    record.trace_id = "trace-1"
    record.span_name = "span"
    record.task_type = "background"
    record.extra_json = {"ok": True}
    record.extra_unserializable = object()

    data = handler._format_log_record(record)

    assert data["trace_id"] == "trace-1"
    assert data["span_name"] == "span"
    assert data["task_type"] == "background"
    assert data["extra"]["extra_json"] == {"ok": True}
    assert "extra_unserializable" not in data["extra"]


def test_format_log_record_truncates_large_message_and_traceback():
    handler = WebSocketLogHandler(FakeManager(), ["hivememory.*"])
    formatter = logging.Formatter()
    handler.setFormatter(formatter)
    try:
        raise RuntimeError("x" * (MAX_TRACEBACK_LENGTH + 100))
    except RuntimeError:
        record = logging.getLogger("hivememory.patchouli").makeRecord(
            "hivememory.patchouli",
            logging.ERROR,
            __file__,
            1,
            "m" * (MAX_MESSAGE_LENGTH + 10),
            (),
            exc_info=sys.exc_info(),
        )

    data = handler._format_log_record(record)

    assert data["message"].endswith("... [truncated]")
    assert data["exception"]["traceback"].endswith("\n... [truncated]")


@pytest.mark.asyncio
async def test_emit_schedules_broadcast_inside_running_loop():
    manager = FakeManager()
    handler = WebSocketLogHandler(manager, ["hivememory.*"])

    handler.emit(_record())
    await asyncio.sleep(0)

    assert manager.messages[0]["message"] == "hello"


def test_emit_skips_non_error_when_rate_limited():
    manager = FakeManager()
    handler = WebSocketLogHandler(manager, ["hivememory.*"], max_rate=0)

    handler.emit(_record(level=logging.INFO))

    assert manager.messages == []
