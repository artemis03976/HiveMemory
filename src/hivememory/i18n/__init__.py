"""HiveMemory i18n 基础设施。"""

from hivememory.i18n.types import (
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    Language,
    normalize_language,
)
from hivememory.i18n.resolver import resolve_language
from hivememory.i18n.memory_compiler import (
    get_memory_atom_text,
    get_memory_envelope_text,
    get_memory_section_title,
)
from hivememory.i18n.time_formatter import get_time_formatter_text

__all__ = [
    "DEFAULT_LANGUAGE",
    "FALLBACK_LANGUAGE",
    "Language",
    "get_memory_atom_text",
    "get_memory_envelope_text",
    "get_memory_section_title",
    "get_time_formatter_text",
    "normalize_language",
    "resolve_language",
]
