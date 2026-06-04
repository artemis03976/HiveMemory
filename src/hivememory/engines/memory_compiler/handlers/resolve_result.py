"""ResolveResult text helpers for target-first handlers."""

from __future__ import annotations

from hivememory.i18n.memory_compiler import get_resolve_result_text


def _t(key: str, language: str | None = None) -> str:
    return get_resolve_result_text(key, language)
