"""
HiveMemory 时间格式化工具。

提供多语言支持的相对时间格式化。
"""

from datetime import datetime
from typing import Optional

from hivememory.i18n import Language, get_time_formatter_text, resolve_language


class TimeFormatter:
    """
    把 datetime 对象格式化为人类可读相对时间字符串的工具类。

    支持多语言与可自定义的陈旧记忆警告。

    Features:
    - 双语支持（英文/中文）
    - 可配置的陈旧警告阈值
    - 灵活的时间单位（月/天/小时/刚刚）

    Example:
        >>> from datetime import timedelta
        >>> formatter = TimeFormatter(language=Language.ZH)
        >>> formatter.format(datetime.now() - timedelta(days=5))
        '5 天前'
        >>> formatter.format(datetime.now() - timedelta(days=100))
        '3 个月前 (警告：陈旧)'
        >>> formatter_en = TimeFormatter(language=Language.EN)
        >>> formatter_en.format(datetime.now() - timedelta(days=5))
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

    def format(self, dt: datetime, reference: Optional[datetime] = None) -> str:
        """
        把 datetime 格式化为相对时间字符串。

        Args:
            dt: 要格式化的 datetime
            reference: 参考时间（默认取当前时间）

        Returns:
            格式化后的相对时间字符串，例如 "5 天前" 或 "2 months ago"
        """
        dt, reference = self._normalize_datetimes(dt=dt, reference=reference)

        delta = reference - dt
        total_days = delta.days

        if total_days >= self.MONTH_THRESHOLD:
            months = total_days // self.MONTH_THRESHOLD
            result = self._text("months_ago").format(months=months)
            if total_days > self.stale_days:
                result += self._text("stale_warning")
            return result
        elif total_days > 0:
            return self._text("days_ago").format(days=total_days)
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return self._text("hours_ago").format(hours=hours)
        else:
            return self._text("recently")

    def _text(self, key: str) -> str:
        return get_time_formatter_text(key, self.language)

    @staticmethod
    def _normalize_datetimes(
        dt: datetime, reference: Optional[datetime]
    ) -> tuple[datetime, datetime]:
        """
        归一化 datetime 的时区感知，避免 naive 与 aware 相减报错。

        Rules:
        - 若 `reference` 为 None，则以与 `dt` 相同的感知性推导它。
        - 若一方 naive 另一方 aware，把 naive 一方对齐到 aware 一方的 tzinfo。
        """
        if reference is None:
            if dt.tzinfo is not None:
                return dt, datetime.now(dt.tzinfo)
            return dt, datetime.now()

        dt_is_aware = dt.tzinfo is not None
        ref_is_aware = reference.tzinfo is not None

        if dt_is_aware and not ref_is_aware:
            return dt, reference.replace(tzinfo=dt.tzinfo)
        if not dt_is_aware and ref_is_aware:
            return dt.replace(tzinfo=reference.tzinfo), reference
        return dt, reference


def format_time_ago(
    dt: datetime,
    language: str | Language | None = None,
    stale_days: int = TimeFormatter.DEFAULT_STALE_DAYS,
    reference: Optional[datetime] = None,
) -> str:
    """
    把 datetime 格式化为相对时间的快捷函数。

    Args:
        dt: 要格式化的 datetime
        language: 输出语言（默认中文）
        stale_days: 显示陈旧警告的天数阈值（默认 90）
        reference: 参考时间（默认取当前时间）

    Returns:
        格式化后的相对时间字符串

    Example:
        >>> from datetime import timedelta
        >>> format_time_ago(datetime.now() - timedelta(days=5))
        '5 天前'
        >>> format_time_ago(datetime.now() - timedelta(days=5), language=Language.ENGLISH)
        '5 days ago'
    """
    formatter = TimeFormatter(language=language, stale_days=stale_days)
    return formatter.format(dt, reference=reference)
