"""HiveMemory i18n 基础设施。"""

from hivememory.i18n.types import (
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    Language,
    normalize_language,
)
from hivememory.i18n.resolver import (
    get_default_language,
    resolve_language,
    set_default_language,
)
from hivememory.i18n.memory_compiler import (
    get_memory_atom_text,
    get_memory_envelope_text,
    get_memory_section_title,
    get_pending_atom_text,
    get_resolve_result_text,
)
from hivememory.i18n.prompts import (
    get_generation_prompt_text,
    get_gateway_prompt_text,
    get_mtp_prompt_text,
    get_mtp_verb_text,
    get_relay_prompt_text,
    get_system_prompt_text,
)
from hivememory.i18n.time_formatter import get_time_formatter_text

from hivememory.i18n.mtp_runtime import (
    get_mtp_error_text,
    get_mtp_info_text,
    get_mtp_warning_text,
)
from hivememory.i18n.syscall_runtime import (
    get_syscall_error_text,
    get_syscall_info_text,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "FALLBACK_LANGUAGE",
    "Language",
    "get_memory_atom_text",
    "get_memory_envelope_text",
    "get_memory_section_title",
    "get_pending_atom_text",
    "get_resolve_result_text",
    "get_generation_prompt_text",
    "get_gateway_prompt_text",
    "get_mtp_prompt_text",
    "get_mtp_verb_text",
    "get_relay_prompt_text",
    "get_system_prompt_text",
    "get_time_formatter_text",
    "get_mtp_error_text",
    "get_mtp_info_text",
    "get_mtp_warning_text",
    "get_syscall_error_text",
    "get_syscall_info_text",
    "get_default_language",
    "normalize_language",
    "resolve_language",
    "set_default_language",
]
