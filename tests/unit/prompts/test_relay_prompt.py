from hivememory.prompts.relay import get_relay_system_prompt


def test_relay_system_prompt_default_chinese():
    prompt = get_relay_system_prompt()

    assert "[系统指令]" in prompt
    assert "状态快照" in prompt
    assert "### 1. 核心目标" in prompt
    assert "### 4. 当前焦点" in prompt


def test_relay_system_prompt_english():
    prompt = get_relay_system_prompt(language="en")

    assert "[System Instruction]" in prompt
    assert "State Snapshot" in prompt
    assert "### 1. Objective" in prompt
    assert "### 4. Current Focus" in prompt


def test_relay_system_prompt_language_alias():
    prompt = get_relay_system_prompt(language="en-US")

    assert "[System Instruction]" in prompt
    assert "### 1. Objective" in prompt
