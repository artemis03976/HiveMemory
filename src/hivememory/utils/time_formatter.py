"""
Time formatting utilities for HiveMemory.

Provides relative time formatting with multilingual support.
"""

from datetime import datetime
from typing import Optional

from hivememory.i18n import Language, get_time_formatter_text, resolve_language


class TimeFormatter:
    """
    Utility class for formatting datetime objects into human-readable relative time strings.

    Supports multiple languages and customizable stale memory warnings.

    Features:
    - Bilingual support (English/Chinese)
    - Configurable stale warning threshold
    - Flexible time units (months, days, hours, recent)

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

    # Threshold constants (days)
    MONTH_THRESHOLD = 30
    DEFAULT_STALE_DAYS = 90

    def __init__(
        self,
        language: str | Language | None = None,
        stale_days: int = DEFAULT_STALE_DAYS,
    ):
        """
        Initialize the TimeFormatter.

        Args:
            language: The language for output strings (default: global fallback)
            stale_days: Number of days after which a memory is considered stale (default: 90)
        """
        self.language = resolve_language(explicit=language)
        self.stale_days = stale_days

    def format(self, dt: datetime, reference: Optional[datetime] = None) -> str:
        """
        Format a datetime as a relative time string.

        Args:
            dt: The datetime to format
            reference: Reference datetime (defaults to current time)

        Returns:
            Formatted relative time string, e.g., "5 天前" or "2 months ago"
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
    def _normalize_datetimes(dt: datetime, reference: Optional[datetime]) -> tuple[datetime, datetime]:
        """
        Normalize datetime timezone awareness to avoid naive/aware subtraction errors.

        Rules:
        - If `reference` is None, derive it with the same awareness as `dt`.
        - If one is naive and the other is aware, align the naive one to the aware one's tzinfo.
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
    Quick function to format a datetime as relative time.

    Args:
        dt: The datetime to format
        language: The language for output (default: Chinese)
        stale_days: Days before showing stale warning (default: 90)
        reference: Reference datetime (defaults to current time)

    Returns:
        Formatted relative time string

    Example:
        >>> from datetime import timedelta
        >>> format_time_ago(datetime.now() - timedelta(days=5))
        '5 天前'
        >>> format_time_ago(datetime.now() - timedelta(days=5), language=Language.ENGLISH)
        '5 days ago'
    """
    formatter = TimeFormatter(language=language, stale_days=stale_days)
    return formatter.format(dt, reference=reference)
