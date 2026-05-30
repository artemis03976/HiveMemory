"""TimeFormatter i18n text."""

from __future__ import annotations

from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.types import Language


_TIME_FORMATTER_TEXT_ZH = {
    "months_ago": "{months} 个月前",
    "days_ago": "{days} 天前",
    "hours_ago": "{hours} 小时前",
    "recently": "最近",
    "stale_warning": " (警告：陈旧)",
}

_TIME_FORMATTER_TEXT_EN = {
    "months_ago": "{months} months ago",
    "days_ago": "{days} days ago",
    "hours_ago": "{hours} hours ago",
    "recently": "recently",
    "stale_warning": " (Warning: Old)",
}


def get_time_formatter_text(key: str, language: str | Language | None = None) -> str:
    """Return a TimeFormatter text fragment."""
    resolved = resolve_language(explicit=language)
    texts = _TIME_FORMATTER_TEXT_EN if resolved == Language.EN else _TIME_FORMATTER_TEXT_ZH
    try:
        return texts[key]
    except KeyError as exc:
        raise KeyError(f"Unknown time formatter i18n key: {key}") from exc


__all__ = ["get_time_formatter_text"]
