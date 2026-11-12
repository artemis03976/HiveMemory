"""HiveMemory 的语言设置与语言解析。

HiveMemory 只有一个语言设置：组合根装配时按配置 ``i18n.language`` 设置一次，提示词、MCP 文字与
HTTP 错误说明都跟随它。调用方显式给出的语言（例如执行帧携带的语言）优先于这个设置。
"""

from __future__ import annotations

from hivememory.i18n.types import DEFAULT_LANGUAGE, Language, normalize_language

_language: Language = DEFAULT_LANGUAGE


def set_language(language: Language) -> None:
    """设置进程的语言；只接受 ``zh`` 与 ``en``，其他值抛 ``ValueError``。"""
    global _language
    _language = Language(language)


def get_language() -> Language:
    """返回进程的语言设置。"""
    return _language


def resolve_language(*, explicit: str | Language | None = None) -> Language:
    """给出可识别的显式语言时用它，否则用语言设置。"""
    return normalize_language(explicit) or _language
