"""utils.time 单元测试：utc_now 与 require_utc 的核心契约。"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from hivememory.utils.time import require_utc, utc_now


def test_utc_now_returns_aware_utc_datetime():
    """utc_now 返回携带 UTC 时区的 aware datetime。"""
    value = utc_now()

    assert value.tzinfo is not None
    assert value.utcoffset() == timedelta(0)
    assert value.tzinfo is UTC or value.utcoffset() == UTC.utcoffset(value)


def test_utc_now_monotonic_between_calls():
    """连续取时不回退（同一进程内第二次调用不早于第一次）。"""
    first = utc_now()
    second = utc_now()

    assert second >= first


def test_require_utc_normalizes_offset_to_utc():
    """带 offset 的 aware 值规范化为 UTC 等价时点。"""
    cst = timezone(timedelta(hours=8))
    value = datetime(2026, 9, 22, 20, 0, 0, tzinfo=cst)

    normalized = require_utc(value)

    assert normalized.utcoffset() == timedelta(0)
    assert normalized == datetime(2026, 9, 22, 12, 0, 0, tzinfo=UTC)


def test_require_utc_rejects_naive_value():
    """naive 值无法解释来源时区，fail closed 并给出可诊断信息。"""
    naive = datetime(2026, 9, 22, 12, 0, 0)

    with pytest.raises(ValueError, match="naive"):
        require_utc(naive)


def test_require_utc_does_not_mutate_input():
    """规范化返回新值，不修改原对象的 tzinfo。"""
    cst = timezone(timedelta(hours=8))
    value = datetime(2026, 9, 22, 20, 0, 0, tzinfo=cst)

    normalized = require_utc(value)

    assert value.tzinfo is cst
    assert normalized is not value
