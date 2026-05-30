from hivememory.i18n import get_generation_prompt_text


def test_generation_prompt_getter_supports_three_modes_default_chinese():
    passive_system = get_generation_prompt_text("passive", "system_prompt")
    passive_user = get_generation_prompt_text("passive", "user_prompt")
    write_system = get_generation_prompt_text("write", "system_prompt")
    write_user = get_generation_prompt_text("write", "user_prompt")
    update_system = get_generation_prompt_text("update", "system_prompt")
    update_user = get_generation_prompt_text("update", "user_prompt")

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
