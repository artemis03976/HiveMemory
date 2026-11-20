import pytest

from hivememory.i18n import get_generation_prompt_text


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
    """修订的输出格式在提示词中手写，系统提示词的 JSON 示例与用户提示词的输出要求都带引用。"""
    system = get_generation_prompt_text("update", "system_prompt", language)
    user = get_generation_prompt_text("update", "user_prompt", language)

    assert '"source_segments": ["2.3"]' in system
    assert '"source_segments": [...]' in user


@pytest.mark.parametrize("language", ["zh", "en"])
def test_librarian_prompt_orders_the_three_steps_and_gives_the_action_format(language):
    """工作顺序依次是列出本话题的写入、检索库、写入、结束；动作是单个 JSON 对象，
    步数与检索条数的上限随配置印出。"""
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
        order.index("list_topic_writes")
        < order.index("search_memories")
        < order.index("write_memory")
        < order.index("finish")
    )
    assert '{"tool": "' in prompt
    assert [text for text in budget if text not in prompt] == []
