"""中立提示词编译的人格、环境通知、空段与语言选择。"""

import pytest

from hivememory.core.models import AgentProfile
from hivememory.i18n import get_default_language, set_default_language
from hivememory.prompts.compilation import compile_prompt_input


@pytest.mark.parametrize(
    "language,expected",
    [
        (
            "zh",
            "[系统通知] 记忆存储当前离线。依赖记忆存储的操作将失败。"
            "请使用已有上下文继续对话。\n\n### 角色设定 ###\n\nPERSONA",
        ),
        (
            "en",
            "[SYSTEM NOTICE] Memory storage is currently OFFLINE. "
            "Operations that require memory storage will fail. "
            "Continue using the available context.\n\n### PERSONA ###\n\nPERSONA",
        ),
    ],
)
def test_offline_notice_precedes_persona_without_cpu_configuration(language, expected):
    """离线通知不依赖工具开关，先于人格且不承诺执行者的工具能力。"""
    compiled = compile_prompt_input(
        agent_profile=AgentProfile(persona="PERSONA", language=language),
        storage_available=False,
        memory_context="",
        attachment_context="",
    )
    assert compiled.instructions == expected
    assert compiled.language == language


def test_online_empty_persona_and_materials_produce_empty_blocks():
    """在线且没有人格或材料时，不添加空标题、占位内容或连接符。"""
    compiled = compile_prompt_input(
        agent_profile=AgentProfile(persona=""),
        storage_available=True,
        memory_context="",
        attachment_context="",
    )
    assert (compiled.instructions, compiled.turn_context) == ("", "")


@pytest.mark.parametrize("profile_language,expected", [("en-US", "en"), ("", "zh")])
def test_compilation_resolves_profile_language_before_process_default(profile_language, expected):
    """Profile 语言别名优先，未指定语言才使用进程默认值。"""
    previous = get_default_language()
    try:
        set_default_language("zh")
        compiled = compile_prompt_input(
            agent_profile=AgentProfile(persona="PERSONA", language=profile_language),
            storage_available=True,
            memory_context="",
            attachment_context="",
        )
    finally:
        set_default_language(previous)
    assert compiled.language == expected
    assert compiled.instructions == (
        "### PERSONA ###\n\nPERSONA" if expected == "en" else "### 角色设定 ###\n\nPERSONA"
    )
