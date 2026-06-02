import re

from hivememory.agent_runtime.syscalls.clock import sys_clock


class TestSysClock:
    """sys_clock 函数直接测试"""

    def test_default_format(self):
        result = sys_clock({})
        assert re.match(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \(UTC[+-]\d+\)",
            result,
        ), f"Unexpected format: {result}"

    def test_iso_format(self):
        result = sys_clock({"format": "iso"})
        assert "T" in result
        assert "+" in result or "-" in result or "Z" in result

    def test_date_format(self):
        result = sys_clock({"format": "date"})
        assert re.match(r"\d{4}-\d{2}-\d{2}$", result)

    def test_time_format(self):
        result = sys_clock({"format": "time"})
        assert re.match(r"\d{2}:\d{2}:\d{2}$", result)

    def test_no_args_uses_default(self):
        result = sys_clock({})
        assert "UTC" in result

    def test_unknown_format_uses_default(self):
        result = sys_clock({"format": "unknown"})
        assert "UTC" in result
