from hivememory.i18n import get_relay_prompt_text


def test_relay_system_prompt_default_chinese():
    prompt = get_relay_prompt_text("system_prompt")

    assert "[系统指令]" in prompt
    assert "状态快照" in prompt
    assert "### 1. 核心目标" in prompt
    assert "### 4. 当前焦点" in prompt


def test_relay_system_prompt_english():
    prompt = get_relay_prompt_text("system_prompt", language="en")

    assert "[System Instruction]" in prompt
    assert "State Snapshot" in prompt
    assert "### 1. Objective" in prompt
    assert "### 4. Current Focus" in prompt


def test_relay_system_prompt_language_alias():
    prompt = get_relay_prompt_text("system_prompt", language="en-US")

    assert "[System Instruction]" in prompt
    assert "### 1. Objective" in prompt


def test_relay_user_prompt_template():
    prompt = get_relay_prompt_text("user_prompt").format(
        previous_summary="old state",
        recent_events="new events",
    )

    assert "<old_state_summary>" in prompt
    assert "old state" in prompt
    assert "<recent_events>" in prompt
    assert "new events" in prompt


def test_relay_previous_summary_empty():
    assert get_relay_prompt_text("previous_summary_empty") == "无。当前为新话题。"
    assert get_relay_prompt_text("previous_summary_empty", language="en") == "None. This is a new topic."
