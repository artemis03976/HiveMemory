"""
HiveMemory 时间格式化工具。

提供多语言支持的相对时间格式化。

输入契约（A2-P 时间边界）：``format`` 的 ``dt`` 与显式 ``reference`` 必须是
timezone-aware datetime，并在入口经 :func:`require_utc` 规范化为 UTC；naive
值不再猜测时区，直接抛出 ``ValueError``。需要确定性结果的调用方（如
MemoryCompiler 与测试）应显式传入同一 ``reference``。本工具是纯展示组件：
不保存当前时间、不选择业务时间字段、也不实现任何时钟。
"""

from datetime import datetime

from hivememory.i18n import Language, get_time_formatter_text, resolve_language
from hivememory.utils.time import require_utc, utc_now


class TimeFormatter:
    """
    把 datetime 对象格式化为人类可读相对时间字符串的工具类。

    支持多语言与可自定义的陈旧记忆警告。

    Features:
    - 双语支持（英文/中文）
    - 可配置的陈旧警告阈值
    - 灵活的时间单位（月/天/小时/刚刚）

    Example:
        >>> from datetime import UTC, datetime, timedelta
        >>> formatter = TimeFormatter(language=Language.ZH)
        >>> now = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
        >>> formatter.format(now - timedelta(days=5), reference=now)
        '5 天前'
        >>> formatter.format(now - timedelta(days=100), reference=now)
        '3 个月前 (警告：陈旧)'
        >>> formatter_en = TimeFormatter(language=Language.EN)
        >>> formatter_en.format(now - timedelta(days=5), reference=now)
        '5 days ago'
    """

    # 阈值常量（单位：天）
    MONTH_THRESHOLD = 30
    DEFAULT_STALE_DAYS = 90

    def __init__(
        self,
        language: str | Language | None = None,
        stale_days: int = DEFAULT_STALE_DAYS,
    ):
        """
        初始化 TimeFormatter。

        Args:
            language: 输出文案语言（默认：全局回退语言）
            stale_days: 记忆超过多少天视为陈旧（默认 90）
        """
        self.language = resolve_language(explicit=language)
        self.stale_days = stale_days

    def format(self, dt: datetime, reference: datetime | None = None) -> str:
        """
        把 datetime 格式化为相对时间字符串。

        Args:
            dt: 要格式化的 timezone-aware datetime（naive 值拒绝）
            reference: 参考时间（timezone-aware；默认取当前 UTC 时间）

        Returns:
            格式化后的相对时间字符串，例如 "5 天前" 或 "2 months ago"

        Raises:
            ValueError: ``dt`` 或 ``reference`` 是 naive datetime
        """
        dt = require_utc(dt)
        reference = require_utc(reference) if reference is not None else utc_now()

        delta = reference - dt
        total_seconds = delta.total_seconds()

        # 未来时间与不满一小时的时间差统一显示为"最近"，不依赖
        # timedelta.seconds 对负时间差的折叠行为。
        if total_seconds < 3600:
            return self._text("recently")

        total_hours = int(total_seconds // 3600)
        total_days = int(total_seconds // 86400)

        if total_days >= self.MONTH_THRESHOLD:
            months = total_days // self.MONTH_THRESHOLD
            result = self._text("months_ago").format(months=months)
            if total_days > self.stale_days:
                result += self._text("stale_warning")
            return result
        if total_days > 0:
            return self._text("days_ago").format(days=total_days)
        return self._text("hours_ago").format(hours=total_hours)

    def _text(self, key: str) -> str:
        return get_time_formatter_text(key, self.language)


def format_time_ago(
    dt: datetime,
    language: str | Language | None = None,
    stale_days: int = TimeFormatter.DEFAULT_STALE_DAYS,
    reference: datetime | None = None,
) -> str:
    """
    把 datetime 格式化为相对时间的快捷函数。

    Args:
        dt: 要格式化的 timezone-aware datetime（naive 值拒绝）
        language: 输出语言（默认中文）
        stale_days: 显示陈旧警告的天数阈值（默认 90）
        reference: 参考时间（timezone-aware；默认取当前 UTC 时间）

    Returns:
        格式化后的相对时间字符串

    Example:
        >>> from datetime import UTC, datetime, timedelta
        >>> now = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)
        >>> format_time_ago(now - timedelta(days=5), reference=now)
        '5 天前'
        >>> format_time_ago(now - timedelta(days=5), language=Language.EN, reference=now)
        '5 days ago'
    """
    formatter = TimeFormatter(language=language, stale_days=stale_days)
    return formatter.format(dt, reference=reference)
