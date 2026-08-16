import logging

from hivememory.infrastructure.trace_context import (
    TraceInjectFilter,
    generate_trace_id,
    reset_trace_context,
    set_trace_context,
)


def test_trace_inject_filter_uses_defaults():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)

    assert TraceInjectFilter().filter(record) is True
    assert record.trace_id == "system"
    assert record.span_name == "main"
    assert record.task_type == "foreground"


def test_set_and_reset_trace_context():
    tokens = set_trace_context("trace-1", "span-a", "background")
    try:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)
        TraceInjectFilter().filter(record)
        assert record.trace_id == "trace-1"
        assert record.span_name == "span-a"
        assert record.task_type == "background"
    finally:
        # 无论断言是否失败都恢复全局 context，避免污染后续测试
        reset_trace_context(tokens)

    restored = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)
    TraceInjectFilter().filter(restored)
    assert restored.trace_id == "system"
    assert restored.span_name == "main"
    assert restored.task_type == "foreground"


def test_generate_trace_id_uses_optional_prefix():
    trace_id = generate_trace_id("chat")

    prefix, suffix = trace_id.split("-", 1)
    assert prefix == "chat"
    assert len(suffix) == 8
    assert len(generate_trace_id()) == 8
