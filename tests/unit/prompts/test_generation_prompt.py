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
@pytest.mark.parametrize("mode", ["passive", "write", "update"])
def test_generation_prompts_explain_action_section_and_forbid_credentials(mode, language):
    """转写带有思考与外部工具动作的提示词，必须说明动作段的含义与凭据约束。"""
    prompt = get_generation_prompt_text(mode, "system_prompt", language)
    credential_rule = {"zh": "不要把密钥、令牌", "en": "Never write credentials"}[language]

    assert "THOUGHT" in prompt
    assert "TOOL / RESULT" in prompt
    assert "chars omitted" in prompt
    assert "OMITTED" in prompt
    assert credential_rule in prompt
