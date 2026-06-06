import re

from hivememory.agent_runtime.mtp.syscalls.clock import sys_clock


class TestSysClock:
    """sys_clock 函数直接测试"""

    def test_default_format(self):
        result = sys_clock({})
        assert result.ok
        assert re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d+\)",
            result.content,
        ), f"Unexpected format: {result.content}"

    def test_iso_format(self):
        result = sys_clock({"format": "iso"})
        assert result.ok
        assert "T" in result.content
        assert "+" in result.content or "-" in result.content or "Z" in result.content

    def test_date_format(self):
        result = sys_clock({"format": "date"})
        assert result.ok
        assert re.match(r"\d{4}-\d{2}-\d{2}$", result.content)

    def test_time_format(self):
        result = sys_clock({"format": "time"})
        assert result.ok
        assert re.match(r"\d{2}:\d{2}:\d{2}$", result.content)

    def test_no_args_uses_default(self):
        result = sys_clock({})
        assert result.ok
        assert "UTC" in result.content

    def test_unknown_format_uses_default(self):
        result = sys_clock({"format": "unknown"})
        assert result.ok
        assert "UTC" in result.content
