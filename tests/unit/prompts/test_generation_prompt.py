import re

import pytest

from hivememory.i18n import get_generation_prompt_text
from hivememory.i18n.prompts import _GENERATION_PROMPT_TEXT_EN, _GENERATION_PROMPT_TEXT_ZH

# CJK 标点、汉字与全角字符；英文提示词中出现任何一个，模型都可能跟着用中文写记忆。
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def test_generation_prompts_have_the_same_keys_in_both_languages():
    """每种生成模式的中英文提示词键齐全，切换语言时不会缺少错误文本或重试文本。"""
    assert {mode: sorted(texts) for mode, texts in _GENERATION_PROMPT_TEXT_EN.items()} == {
        mode: sorted(texts) for mode, texts in _GENERATION_PROMPT_TEXT_ZH.items()
    }


def test_english_generation_prompts_contain_no_chinese():
    leaking = [
        f"{mode}.{key}"
        for mode, texts in _GENERATION_PROMPT_TEXT_EN.items()
        for key, text in texts.items()
        if _CJK.search(text)
    ]

    assert leaking == []


def test_generation_prompt_getter_supports_three_modes_default_chinese():
    passive_system = get_generation_prompt_text("passive", "system_prompt", "zh")
    passive_user = get_generation_prompt_text("passive", "user_prompt", "zh")
    write_system = get_generation_prompt_text("write", "system_prompt", "zh")
    write_user = get_generation_prompt_text("write", "user_prompt", "zh")
    update_system = get_generation_prompt_text("update", "system_prompt", "zh")
    update_user = get_generation_prompt_text("update", "user_prompt", "zh")

    assert "记忆管理员" in passive_system
    assert "{format_instructions}" in passive_system
    assert "{transcript}" in passive_user

    assert "主动响应" in write_system
    assert "{write_content}" in write_user
    assert "{write_reason}" in write_user

    assert "编辑审查" in update_system
    assert "{old_payload}" in update_user
    assert "{instruction}" in update_user


def test_generation_prompt_getter_supports_three_modes_english():
    passive_system = get_generation_prompt_text("passive", "system_prompt", "en")
    write_user = get_generation_prompt_text("write", "user_prompt", "en")
    update_system = get_generation_prompt_text("update", "system_prompt", "en")

    assert "memory manager" in passive_system
    assert "{format_instructions}" in passive_system
    assert "Agent-submitted Memory Draft" in write_user
    assert "{write_content}" in write_user
    assert "Editor Mode" in update_system


def test_generation_prompt_getter_supports_language_alias():
    prompt = get_generation_prompt_text("update", "user_prompt", "en-US")

    assert "Target Memory" in prompt
    assert "{memory_title}" in prompt


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("mode", ["passive", "write", "update", "librarian"])
def test_generation_prompts_explain_process_entries_and_forbid_credentials(mode, language):
    """转写中助手正文与过程条目按顺序穿插，提示词必须说明这一格式、各条目的含义与凭据约束。"""
    prompt = get_generation_prompt_text(mode, "system_prompt", language)
    credential_rule = {"zh": "不要把密钥、令牌", "en": "Never write credentials"}[language]

    assert "[Assistant]" in prompt
    assert "THOUGHT" in prompt
    assert "TOOL / RESULT" in prompt
    assert "chars omitted" in prompt
    assert credential_rule in prompt


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("mode", ["passive", "write", "update", "librarian"])
def test_generation_prompts_explain_segment_labels_and_citations(mode, language):
    """材料在每段之前印出段标签，生成材料的各类提示词都说明标签的写法与引用字段。"""
    prompt = get_generation_prompt_text(mode, "system_prompt", language)

    assert "[Segment 2.3]" in prompt
    assert "source_segments" in prompt


@pytest.mark.parametrize("language", ["zh", "en"])
def test_update_prompt_handwritten_format_includes_citations(language):
    """修订的输出格式是手写的编辑集：系统提示词的 JSON 示例与用户提示词的输出要求都带编辑
    与引用，用户提示词给出目标记忆的摘要与标签。"""
    system = get_generation_prompt_text("update", "system_prompt", language)
    user = get_generation_prompt_text("update", "user_prompt", language)

    assert '"source_segments": ["2.3"]' in system
    assert '"op": "replace"' in system
    assert '"source_segments": [...]' in user
    assert '"edits": [...]' in user
    assert ["{memory_summary}" in user, "{memory_tags}" in user] == [True, True]


@pytest.mark.parametrize("language", ["zh", "en"])
def test_librarian_prompt_orders_the_three_steps_and_prints_the_limits(language):
    """工作顺序依次是列出本话题生成的记忆、检索库、修订、新建、结束；步数与检索条数的上限
    随配置印出。"""
    prompt = get_generation_prompt_text("librarian", "system_prompt", language).format(
        max_steps=7, search_top_k=3
    )
    heading = {"zh": "## 工作顺序", "en": "## Order of Work"}[language]
    start = prompt.index(heading)
    order = prompt[start : prompt.index("##", start + len(heading))]
    budget = {
        "zh": ["最多有 7 步", "最多 3 条"],
        "en": ["at most 7 steps", "at most 3 memories"],
    }[language]

    assert (
        order.index("list_generated_memories")
        < order.index("search_memories")
        < order.index("revise_memory")
        < order.index("write_memory")
        < order.index("finish")
    )
    assert [text for text in budget if text not in prompt] == []
