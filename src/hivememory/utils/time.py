"""UTC 业务时间工具（A2-P 时间边界计划的唯一生产时间入口）。

只提供两个纯函数入口，不保存任何可变的"当前时间"状态；需要可控时间的
服务在各自构造函数上注入 ``Callable[[], datetime]``，测试传入 fake callable。
"""

from datetime import UTC, datetime

__all__ = ["require_utc", "utc_now"]


def utc_now() -> datetime:
    """返回 timezone-aware UTC 当前时间。"""
    return datetime.now(UTC)


def require_utc(value: datetime) -> datetime:
    """校验 ``value`` 是 timezone-aware datetime，并规范化为 UTC 返回。

    naive datetime 无法解释来源时区，直接抛出 ``ValueError``；本工具不为
    naive 值补本地时区或 UTC，时区解释属于各 codec/迁移入口的显式职责。
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"datetime 必须携带时区信息，收到 naive 值: {value!r}")
    return value.astimezone(UTC)
