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


@pytest.mark.asyncio
async def test_emit_filters_by_namespace():
    manager = FakeManager()
    handler = WebSocketLogHandler(manager, ["hivememory.patchouli", "hivememory.alice.*"])

    handler.emit(_record("hivememory.patchouli"))
    handler.emit(_record("hivememory.alice.runtime"))
    handler.emit(_record("other"))
    await asyncio.sleep(0)

    assert len(manager.messages) == 2
    assert all(m["message"] == "hello" for m in manager.messages)


@pytest.mark.asyncio
async def test_emit_includes_trace_context_and_json_extra():
    manager = FakeManager()
    handler = WebSocketLogHandler(manager, ["hivememory.*"])

    record = _record()
    record.trace_id = "trace-1"
    record.span_name = "span"
    record.task_type = "background"
    record.extra_json = {"ok": True}
    record.extra_unserializable = object()

    handler.emit(record)
    await asyncio.sleep(0)

    data = manager.messages[0]
    assert data["trace_id"] == "trace-1"
    assert data["span_name"] == "span"
    assert data["task_type"] == "background"
    assert data["extra"]["extra_json"] == {"ok": True}
    assert "extra_unserializable" not in data["extra"]


@pytest.mark.asyncio
async def test_emit_truncates_large_message_and_traceback():
    manager = FakeManager()
    handler = WebSocketLogHandler(manager, ["hivememory.*"])
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

    handler.emit(record)
    await asyncio.sleep(0)

    data = manager.messages[0]
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
