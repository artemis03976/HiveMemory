"""统一语言解析函数。"""

from __future__ import annotations

from hivememory.i18n.types import DEFAULT_LANGUAGE, Language, normalize_language


_default_language: Language | None = None


def set_default_language(language: str | Language | None) -> Language:
    """Set the process-wide i18n default language."""
    global _default_language
    _default_language = normalize_language(language) or DEFAULT_LANGUAGE
    return _default_language


def get_default_language() -> Language:
    """Return the process-wide i18n default language."""
    return _default_language or DEFAULT_LANGUAGE


def resolve_language(
    *,
    explicit: str | Language | None = None,
    profile_language: str | Language | None = None,
    component_language: str | Language | None = None,
    fallback: Language = DEFAULT_LANGUAGE,
) -> Language:
    """按优先级解析语言，返回确定的 Language 值。

    优先级: explicit > profile_language > component_language > process default > fallback
    """
    for candidate in (explicit, profile_language, component_language):
        resolved = normalize_language(candidate)
        if resolved is not None:
            return resolved
    return _default_language or fallback
