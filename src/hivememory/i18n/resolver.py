"""统一语言解析函数。"""

from __future__ import annotations

from hivememory.i18n.types import DEFAULT_LANGUAGE, Language, normalize_language


def resolve_language(
    *,
    explicit: str | Language | None = None,
    profile_language: str | Language | None = None,
    component_language: str | Language | None = None,
    default_language: str | Language | None = None,
    fallback: Language = DEFAULT_LANGUAGE,
) -> Language:
    """按优先级解析语言，返回确定的 Language 值。

    优先级: explicit > profile_language > component_language > default_language > fallback
    """
    for candidate in (explicit, profile_language, component_language, default_language):
        resolved = normalize_language(candidate)
        if resolved is not None:
            return resolved
    return fallback
