"""折叠摘要的提示词与渲染文案：两种语言的键与占位符一致，英文文案里没有中文，按语言设置取用。"""

import re
from string import Formatter

from hivememory.i18n import get_fold_summary_prompt_text
from hivememory.i18n.prompts import _FOLD_SUMMARY_PROMPT_TEXT_EN, _FOLD_SUMMARY_PROMPT_TEXT_ZH

# CJK 标点、汉字与全角字符；英文文案中出现任何一个，模型都可能跟着用中文写摘要。
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def _placeholders(text: str) -> list[str]:
    return sorted(name for _, name, _, _ in Formatter().parse(text) if name)


def test_both_languages_have_the_same_keys_and_placeholders():
    """切换语言时每一段文案都在，填入的字段也相同，渲染这一次的摘要不会缺键或缺字段。"""
    assert {key: _placeholders(text) for key, text in _FOLD_SUMMARY_PROMPT_TEXT_EN.items()} == {
        key: _placeholders(text) for key, text in _FOLD_SUMMARY_PROMPT_TEXT_ZH.items()
    }


def test_english_texts_contain_no_chinese():
    leaking = [key for key, text in _FOLD_SUMMARY_PROMPT_TEXT_EN.items() if _CJK.search(text)]

    assert leaking == []


def test_language_setting_selects_the_texts():
    assert get_fold_summary_prompt_text("title_turns") == "## 第 {first}–{last} 轮"
    assert get_fold_summary_prompt_text("title_turns", language="en-US") == (
        "## Turns {first}–{last}"
    )
