"""
Time formatting utilities for HiveMemory.

Provides relative time formatting with multilingual support.
"""

from datetime import datetime
from enum import Enum
from typing import Optional


class Language(str, Enum):
    """Supported languages for time formatting."""

    ENGLISH = "en"
    CHINESE = "zh"


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
        >>> formatter = TimeFormatter(language=Language.CHINESE)
        >>> formatter.format(datetime.now() - timedelta(days=5))
        '5 天前'
        >>> formatter.format(datetime.now() - timedelta(days=100))
        '3 个月前 (警告：陈旧)'
        >>> formatter_en = TimeFormatter(language=Language.ENGLISH)
        >>> formatter_en.format(datetime.now() - timedelta(days=5))
        '5 days ago'
    """

    # Translation dictionaries
    TRANSLATIONS = {
        Language.ENGLISH: {
            "months_ago": "{months} months ago",
            "days_ago": "{days} days ago",
            "hours_ago": "{hours} hours ago",
            "recently": "recently",
            "stale_warning": " (Warning: Old)",
        },
        Language.CHINESE: {
            "months_ago": "{months} 个月前",
            "days_ago": "{days} 天前",
            "hours_ago": "{hours} 小时前",
            "recently": "最近",
            "stale_warning": " (警告：陈旧)",
        },
    }

    # Threshold constants (days)
    MONTH_THRESHOLD = 30
    DEFAULT_STALE_DAYS = 90

    def __init__(
        self,
        language: Language = Language.CHINESE,
        stale_days: int = DEFAULT_STALE_DAYS,
    ):
        """
        Initialize the TimeFormatter.

        Args:
            language: The language for output strings (default: Chinese)
            stale_days: Number of days after which a memory is considered stale (default: 90)
        """
        self.language = language
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
        if reference is None:
            reference = datetime.now()

        delta = reference - dt
        total_days = delta.days

        # Get translations for the current language
        t = self.TRANSLATIONS[self.language]

        if total_days >= self.MONTH_THRESHOLD:
            months = total_days // self.MONTH_THRESHOLD
            result = t["months_ago"].format(months=months)
            if total_days > self.stale_days:
                result += t["stale_warning"]
            return result
        elif total_days > 0:
            return t["days_ago"].format(days=total_days)
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            return t["hours_ago"].format(hours=hours)
        else:
            return t["recently"]


def format_time_ago(
    dt: datetime,
    language: Language = Language.CHINESE,
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
