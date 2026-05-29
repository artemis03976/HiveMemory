"""i18n 类型定义与语言归一化。"""

from __future__ import annotations

from enum import StrEnum


class Language(StrEnum):
    ZH = "zh"
    EN = "en"


DEFAULT_LANGUAGE = Language.ZH
FALLBACK_LANGUAGE = Language.EN

_LANGUAGE_ALIASES: dict[str, Language] = {
    "zh": Language.ZH,
    "zh-cn": Language.ZH,
    "zh-hans": Language.ZH,
    "cn": Language.ZH,
    "chinese": Language.ZH,
    "en": Language.EN,
    "en-us": Language.EN,
    "en-gb": Language.EN,
    "english": Language.EN,
}


def normalize_language(value: str | Language | None) -> Language | None:
    """将语言输入归一化为 Language 枚举，未知值返回 None。"""
    if value is None:
        return None
    if isinstance(value, Language):
        return value
    key = str(value).strip().lower()
    if not key:
        return None
    return _LANGUAGE_ALIASES.get(key)
